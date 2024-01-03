import torch
import os
import numpy as np
from models.critic_models import Critic, CriticConv
from models.actor_models import Actor
import torch.nn as nn
import torch.optim as optim
from loss_functions.sarsa_loss_ac import sarsa_loss_ac
from loss_functions.actor_loss import actor_loss
from loss_functions.ppo_loss import ppo_actor_loss
from templates.trainer import Trainer
import torch.multiprocessing as mp
from game.state_scrambling import mild_continuous_scramble
from config import config

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

USE_PPO = True
PPO_EPSILON = 0.1
ACTOR_EPOCHS = 4
SCRAMBLE_PROB = 0.5


class ACTrainer(Trainer):
    def __init__(self, critic_info, actor_info,
                 central_actor=None,
                 central_critic=None,
                 save_name='',
                 save_directory='',
                 qnet=None, actor=None):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm.
        """

        board_size = config.BOARD_SIZE
        start_walls = config.NUMBER_OF_WALLS
        decrease_epsilon_every = config.DECREASE_EPSILON_EVERY
        games_per_iter = config.WORKER_GAMES_BETWEEN_TRAINS
        lambd = config.LAMBD
        gamma = config.GAMMA
        random_proportion = config.RANDOM_PROPORTION
        iterations_only_actor_train = config.ITERATIONS_ONLY_ACTOR_TRAIN
        convolutional = config.USE_CONV_NET
        learning_rate = config.CRITIC_LEARNING_RATE
        epsilon_decay = config.EPSILON_DECAY
        epsilon = config.EPSILON
        minimum_epsilon = config.MINIMUM_EPSILON
        entropy_constant = config.ENTROPY_CONSTANT
        max_grad_norm = config.MAX_GRAD_NORM
        move_prob = config.MOVE_PROB
        minimum_move_prob = config.MINIMUM_MOVE_PROB
        entropy_bias = config.ENTROPY_BIAS
        total_reset_every = config.TOTAL_RESET_EVERY
        cores = config.N_CORES
        old_selfplay = config.OLD_SELFPLAY
        load_from_last = config.LOAD_FROM_LAST
        reload_every = config.RELOAD_EVERY
        reload_distribution = config.RELOAD_DISTRIBUTION

        super().__init__(board_size, start_walls, 4, decrease_epsilon_every,
                         random_proportion, games_per_iter, total_reset_every=total_reset_every, save_name=save_name,
                         cores=cores, old_selfplay=old_selfplay, reload_every=reload_every,
                         save_directory=save_directory)
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

        if old_selfplay:
            self.loaded_actor = Actor(**actor_info).to(device)
            if not convolutional:
                self.loaded_critic = Critic(critic_info['input_dim'], critic_info['critic_size_hidden'],
                                            critic_info['critic_num_hidden']).to(device)
            else:
                self.loaded_critic = CriticConv(**critic_info).to(device)

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
        self.load_from_last = load_from_last
        self.reload_distribution = reload_distribution

        self.global_actor = central_actor
        if self.global_actor is not None:
            self.actor.pull(self.global_actor)
            self.actor_opt = optim.Adam(self.global_actor.parameters(), lr=learning_rate)
        else:
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.global_critic = central_critic
        if self.global_critic is not None:
            self.net.pull(self.global_critic)
            self.optimizer = optim.Adam(self.global_critic.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

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
        w = np.random.uniform()

        # figure out what the random move is
        if u < max([self.epsilon * self.epsilon_decay**decay, self.minimum_epsilon]):
            random_move = True
            if w > SCRAMBLE_PROB:
                if v < max([self.move_prob * self.epsilon_decay**decay, self.minimum_move_prob]):
                    move = np.random.choice(4)
                else:
                    move = np.random.choice(self.bot_out_dimension - 4) + 4
                probability = actor_probabilities[move]
                move = self.possible_moves[move]
            else:
                scrambled_state = mild_continuous_scramble(state)
                scrambled_state_torch = torch.from_numpy(scrambled_state).to(device).float()
                scrambled_actor_move, scrambled_actor_probabilities, scrambled_actor_probability \
                    = self.actor.move(scrambled_state_torch)
                move = scrambled_actor_move
                move_index = np.where(move == 1)[0][0]
                probability = actor_probabilities[move_index]

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
        torch.save(self.net.state_dict(), self.save_directory + '/' + self.save_name + str(j))
        torch.save(self.actor.state_dict(), self.save_directory + '/' + self.save_name + str(j) + 'ACTOR')

    def learn(self, side=None, actor_epochs=ACTOR_EPOCHS):

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

        for epoch in range(actor_epochs):

            # ACTOR
            if not USE_PPO:
                actor_p1_loss, actor_p1_entropy_loss = actor_loss(self.memory_1, advantage_1,
                                                                  entropy_constant=self.entropy_constant,
                                                                  entropy_bias=self.entropy_bias, epoch=epoch)
                actor_p2_loss, actor_p2_entropy_loss = actor_loss(self.memory_2, advantage_2,
                                                                  entropy_constant=self.entropy_constant,
                                                                  entropy_bias=self.entropy_bias, epoch=epoch)

                if side is None:
                    actor_loss_val = actor_p1_loss + actor_p2_loss + actor_p1_entropy_loss + actor_p2_entropy_loss
                elif side == 0:
                    actor_loss_val = actor_p1_loss + actor_p1_entropy_loss
                elif side == 1:
                    actor_loss_val = actor_p2_loss + actor_p2_entropy_loss

            else:
                actor_p1_loss = ppo_actor_loss(self.memory_1, advantage_1, epsilon=PPO_EPSILON, epoch=epoch,
                                               net=self.actor)
                actor_p2_loss = ppo_actor_loss(self.memory_2, advantage_2, epsilon=PPO_EPSILON, epoch=epoch,
                                               net=self.actor)

                if side is None:
                    actor_loss_val = actor_p1_loss + actor_p2_loss
                elif side == 0:
                    actor_loss_val = actor_p1_loss
                elif side == 1:
                    actor_loss_val = actor_p2_loss

            self.actor_opt.zero_grad()
            actor_loss_val.backward()

        # CRITIC
        self.optimizer.zero_grad()
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
            if self.global_actor is not None:
                ensure_shared_grads(self.net, self.global_critic)
            self.optimizer.step()

        """
        if self.global_actor is not None:
            for lp, gp in zip(self.actor.parameters(), self.global_actor.parameters()):
                if gp._grad is not None:
                    gp._grad += lp.grad
                else:
                    gp._grad = lp.grad
        """

        # ACTOR AGAIN
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        if self.global_actor is not None:
            ensure_shared_grads(self.actor, self.global_actor)
        self.actor_opt.step()

        loss = actor_loss_val + critic_loss
        self.learning_iterations_so_far += 1
        return float(loss.detach().cpu().numpy())

    def pull(self):
        self.net.pull(self.global_critic)
        self.actor.pull(self.global_actor)

    def loaded_on_policy_step(self, state, info):
        """
        Function for an old saved version of self to interact with the game
        """
        state_torch = torch.from_numpy(state).to(device).float()
        actor_move, actor_probabilities, actor_probability = self.loaded_actor.move(state_torch)

        random_move = False
        move = actor_move
        probability = actor_probability

        critic_value = self.loaded_critic.feed_forward(state_torch)

        return move, [critic_value, probability, actor_probabilities], random_move

    def load_opponent(self, j=0):
        """
        Chooses an old version of self and loads it in as the opponent
        """

        old_models = os.listdir(self.save_directory + '/saves/')

        if len(old_models) == 0:
            self.loaded_actor.pull(self.actor)
            self.loaded_critic.pull(self.net)
            return

        old_actors = [x for x in old_models if x[-5:] == 'ACTOR']
        prev_savenums = sorted([int(x[4:-5]) for x in old_actors])
        acceptable_choices = prev_savenums[-self.load_from_last:]

        if self.reload_distribution == "uniform":
            choice = np.random.choice(acceptable_choices)
        elif self.reload_distribution == "geometric":
            choice_ind = np.random.geometric(p=0.5)
            if choice_ind >= len(acceptable_choices):
                choice_ind = len(acceptable_choices)
            choice = acceptable_choices[-choice_ind]
        else:
            raise ValueError("invalid distribution")

        """
        print('loading!')
        print(self.save_directory + 'saves/' + self.save_name + str(choice)
                                                     + 'ACTOR')
        """

        self.loaded_actor.load_state_dict(torch.load(self.save_directory + 'saves/' + self.save_name + str(choice)
                                                     + 'ACTOR'))
        self.loaded_critic.load_state_dict(torch.load(self.save_directory + 'saves/' + self.save_name + str(choice)))


class ACWorker(mp.Process, ACTrainer):

    def __init__(self, critic_info, actor_info, central_critic, central_actor, res_q, save_name='', save_directory=''):

        ACTrainer.__init__(self, critic_info=critic_info, actor_info=actor_info, save_name=save_name,
                           save_directory=save_directory, central_actor=central_actor, central_critic=central_critic)

        mp.Process.__init__(self)

        self.iterations = config.EPOCHS_PER_WORKER
        self.res_q = res_q
        self.j = 0

    def run(self):
        self.pull()
        info = self.train(self.iterations, np.inf, '', start_j=self.j, print_every=np.inf)
        self.res_q.put(list(info))


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad