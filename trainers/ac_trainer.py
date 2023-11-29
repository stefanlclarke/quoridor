import torch
import numpy as np
from models.critic_models import Critic, CriticConv
from models.actor_models import Actor
import torch.nn as nn
import torch.optim as optim
from loss_functions.sarsa_loss_ac import sarsa_loss_ac
from loss_functions.actor_loss import actor_loss
from templates.trainer import Trainer
import torch.multiprocessing as mp

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


class ACTrainer(Trainer):
    def __init__(self, board_size, start_walls, critic_info, actor_info, decrease_epsilon_every=100,
                 games_per_iter=100, lambd=0.9, gamma=0.9, random_proportion=0.4,
                 qnet=None, actor=None, iterations_only_actor_train=0, convolutional=False, learning_rate=1e-4,
                 epsilon_decay=0.95, epsilon=0.4, minimum_epsilon=0.05, entropy_constant=1, max_grad_norm=1e5,
                 move_prob=0.4, minimum_move_prob=0.2, entropy_bias=0, save_name='', total_reset_every=np.inf,
                 central_actor=None, central_critic=None, cores=1):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm.
        """

        super().__init__(board_size, start_walls, 4, decrease_epsilon_every,
                         random_proportion, games_per_iter, total_reset_every, save_name=save_name,
                         cores=cores)
        if qnet is None:
            if not convolutional:
                self.net = Critic(critic_info['input_dim'], critic_info['critic_size_hidden'],
                                  critic_info['critic_num_hidden']).to(device)
            else:
                self.net = CriticConv(**critic_info).to(device)
        else:
            self.net = qnet.to(device)
        if actor is None:
            self.actor = Actor(**actor_info).to(device)
        else:
            self.actor = actor.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.learning_iterations_so_far = 0
        self.iterations_only_actor_train = iterations_only_actor_train
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.entropy_constant = entropy_constant
        self.max_grad_norm = max_grad_norm
        self.lambd = lambd
        self.gamma = gamma
        self.move_prob = move_prob
        self.minimum_move_prob = minimum_move_prob
        self.entropy_bias = entropy_bias
        self.save_name = save_name

        self.global_actor = central_actor

        if self.global_actor is not None:
            self.actor.pull(self.global_actor)

        self.global_critic = central_critic
        if self.global_critic is not None:
            self.net.pull(self.global_critic)

    def on_policy_step(self, state, info):
        """
        Handles game decisions and chooses information to save to memory.
        """

        state_torch = torch.from_numpy(state).to(device).float()
        actor_move, actor_probabilities, actor_probability = self.actor.move(state_torch)

        # get the lr decay
        if info is None:
            decay = 1.
        else:
            decay = info[0]

        # decide whether to move randomly
        u = np.random.uniform()
        v = np.random.uniform()

        # figure out what the random move is
        if u < max([self.epsilon * self.epsilon_decay**decay, self.minimum_epsilon]):
            random_move = True
            if v < max([self.move_prob * self.epsilon_decay**decay, self.minimum_move_prob]):
                move = np.random.choice(4)
            else:
                move = np.random.choice(self.bot_out_dimension - 4) + 4
            probability = actor_probabilities[move]
            move = self.possible_moves[move]
        else:
            random_move = False
            move = actor_move
            probability = actor_probability

        critic_value = self.net.feed_forward(state_torch)

        return move, [critic_value, probability, actor_probabilities], random_move

    def off_policy_step(self, state, move_ind, info):
        """
        Chooses information to save to memory when learning off-policy.
        """

        state_torch = torch.from_numpy(state).to(device).float()
        actor_move, actor_probabilities, actor_probability = self.actor.move(state_torch)
        critic_value = self.net.feed_forward(state_torch)

        probability = actor_probability

        probability = actor_probabilities[move_ind]

        return [critic_value, probability, actor_probabilities]

    def save(self, name, info=None):
        """
        Saves network parameters to memory.
        """

        j = info[0]
        torch.save(self.net.state_dict(), self.save_name + str(j))
        torch.save(self.actor.state_dict(), self.save_name + str(j) + 'ACTOR')

    def learn(self):
        if self.learning_iterations_so_far >= self.iterations_only_actor_train:
            train_critic = True
        else:
            train_critic = False

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss_ac(self.memory_1, self.net, 0, self.possible_moves, self.lambd,
                                                    self.gamma, printing=False,
                                                    return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss_ac(self.memory_2, self.net, 0, self.possible_moves, self.lambd,
                                                    self.gamma, printing=False,
                                                    return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        actor_p1_loss, actor_p1_entropy_loss = actor_loss(self.memory_1, advantage_1,
                                                          entropy_constant=self.entropy_constant,
                                                          entropy_bias=self.entropy_bias)
        actor_p2_loss, actor_p2_entropy_loss = actor_loss(self.memory_2, advantage_2,
                                                          entropy_constant=self.entropy_constant,
                                                          entropy_bias=self.entropy_bias)
        actor_loss_val = actor_p1_loss + actor_p2_loss + actor_p1_entropy_loss + actor_p2_entropy_loss

        self.optimizer.zero_grad()
        self.actor_opt.zero_grad()

        if train_critic:
            critic_loss.backward()

            """
            if self.global_critic is not None:
                for lp, gp in zip(self.net.parameters(), self.global_critic.parameters()):
                    if gp._grad is not None:
                        gp._grad += lp.grad
                    else:
                        gp._grad = lp.grad
            """
            self.optimizer.step()

        actor_loss_val.backward()

        """
        if self.global_actor is not None:
            for lp, gp in zip(self.actor.parameters(), self.global_actor.parameters()):
                if gp._grad is not None:
                    gp._grad += lp.grad
                else:
                    gp._grad = lp.grad
        """
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        loss = actor_loss_val + critic_loss
        self.learning_iterations_so_far += 1
        return float(loss.detach().cpu().numpy())

    def pull(self):
        self.net.pull(self.global_critic)
        self.actor.pull(self.global_actor)


class ACWorker(mp.Process, ACTrainer):

    def __init__(self, iterations, board_size, start_walls, critic_info, actor_info, decrease_epsilon_every=100,
                 games_per_iter=100, lambd=0.9, gamma=0.9, random_proportion=0.4,
                 qnet=None, actor=None, iterations_only_actor_train=0, convolutional=False, learning_rate=1e-4,
                 epsilon_decay=0.95, epsilon=0.4, minimum_epsilon=0.05, entropy_constant=1, max_grad_norm=1e5,
                 move_prob=0.4, minimum_move_prob=0.2, entropy_bias=0, save_name='', total_reset_every=np.inf,
                 central_actor=None, central_critic=None, cores=1, res_q=None):

        ACTrainer.__init__(self, board_size, start_walls, critic_info, actor_info, decrease_epsilon_every,
                           games_per_iter, lambd, gamma, random_proportion,
                           qnet, actor, iterations_only_actor_train, convolutional, learning_rate,
                           epsilon_decay, epsilon, minimum_epsilon, entropy_constant, max_grad_norm,
                           move_prob, minimum_move_prob, entropy_bias, save_name, total_reset_every,
                           central_actor, central_critic, cores=1)

        mp.Process.__init__(self)

        self.iterations = iterations
        self.res_q = res_q
        self.j = 0

    def run(self):
        self.pull()
        info = self.train(self.iterations, np.inf, '', start_j=self.j, print_every=np.inf)
        self.res_q.put(list(info))
