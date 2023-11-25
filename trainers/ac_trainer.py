import torch
import numpy as np
from parameters import Parameters
from models.critic_models import Critic, CriticConv
from models.actor_models import Actor
import torch.nn as nn
import torch.optim as optim
from loss_functions.sarsa_loss_ac import sarsa_loss_ac
from loss_functions.actor_loss import actor_loss
from templates.trainer import Trainer

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

parameters = Parameters()
gamma = parameters.gamma
lambd = parameters.lambd
learning_rate = parameters.learning_rate
epsilon = parameters.epsilon
move_prob = parameters.move_prob
minimum_epsilon = parameters.minimum_epsilon
minimum_move_prob = parameters.minimum_move_prob
random_proportion = parameters.random_proportion
entropy_constant = parameters.entropy_constant
max_grad_norm = parameters.max_grad_norm
epsilon = parameters.epsilon
epsilon_decay = parameters.epsilon_decay


class ACTrainer(Trainer):
    def __init__(self, qnet=None, actor=None, iterations_only_actor_train=0, convolutional=False):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm.
        """

        super().__init__(number_other_info=4)
        if qnet is None:
            if not convolutional:
                self.net = Critic().to(device)
            else:
                self.net = CriticConv().to(device)
        else:
            self.net = qnet.to(device)
        if actor is None:
            self.actor = Actor().to(device)
        else:
            self.actor = actor.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.learning_iterations_so_far = 0
        self.iterations_only_actor_train = iterations_only_actor_train

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
        if u < max([epsilon * epsilon_decay**decay, minimum_epsilon]):
            random_move = True
            if v < max([move_prob * epsilon_decay**decay, minimum_move_prob]):
                move = np.random.choice(4)
            else:
                move = np.random.choice(parameters.bot_out_dimension - 4) + 4
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
        torch.save(self.net.state_dict(), './saves/{}'.format(name + str(j)))
        torch.save(self.actor.state_dict(), './saves/{}'.format(name + str(j) + 'ACTOR'))

    def learn(self):
        if self.learning_iterations_so_far >= self.iterations_only_actor_train:
            train_critic = True
        else:
            train_critic = False

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss_ac(self.memory_1, self.net, 0, self.possible_moves, printing=False,
                                                    return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss_ac(self.memory_2, self.net, 0, self.possible_moves, printing=False,
                                                    return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        actor_p1_loss, actor_p1_entropy_loss = actor_loss(self.memory_1, advantage_1,
                                                          entropy_constant=entropy_constant)
        actor_p2_loss, actor_p2_entropy_loss = actor_loss(self.memory_2, advantage_2,
                                                          entropy_constant=entropy_constant)
        actor_loss_val = actor_p1_loss + actor_p2_loss + actor_p1_entropy_loss + actor_p2_entropy_loss

        self.optimizer.zero_grad()
        self.actor_opt.zero_grad()

        if train_critic:
            critic_loss.backward()
            self.optimizer.step()

        actor_loss_val.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_opt.step()

        loss = actor_loss_val + critic_loss
        self.learning_iterations_so_far += 1
        return float(loss.detach().cpu().numpy())
