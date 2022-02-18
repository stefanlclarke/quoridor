import numpy as np
import torch
from templates.trainer import Trainer
from parameters import Parameters
from models.q_models import QNet
from models.wolpertinger_models import WolpActor, k_nearest
import torch.nn as nn
from game.move_reformatter import move_reformatter, unformatted_move_to_index
import copy
import torch.optim as optim
from game.game_helper_functions import *
from loss_functions.sarsa_loss import sarsa_loss
from game.shortest_path import *

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
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion
wolp_k = parameters.wolp_k


class WolpertingerTrainer(Trainer):
    def __init__(self, net=None, actor=None):
        """
        Handles the training of a Q-network using Sarsa Lambda.
        """

        super().__init__(number_other_info=4)
        if net is None:
            self.net = QNet().to(device)
        else:
            self.net = net.to(device)
        if actor is None:
            self.actor = WolpActor()
        else:
            self.actor = actor
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def on_policy_step(self, state, info=None):
        """
        Determines on-policy action and selects information to choose to memory.
        An epsilon-greedy policy is used.
        """

        if info is None:
            decay = 1.
        else:
            decay = info[0]
        #
        actor_action = self.actor.forward(torch.from_numpy(state).to(device).float())
        #actor_action = torch.zeros(parameters.bot_out_dimension)
        moves_to_check = k_nearest(copy.copy(actor_action), wolp_k)
        #moves_to_check = [(x, 0) for x in moves_to_check]#wolp_k]]

        values = []
        for move_ in moves_to_check:
            move = move_[0]
            values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(move)]).to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())
        max_value = values[argmax]
        u = np.random.uniform()
        v = np.random.uniform()
        if u < max([epsilon**decay, minimum_epsilon]):
            random_move = True
            if v < max([move_prob**decay, minimum_move_prob]):
                move = self.possible_moves[np.random.choice(4)]
            else:
                move = self.possible_moves[np.random.choice(parameters.bot_out_dimension - 4) + 4]
            move_value = self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(move)]).to(device).float())
        else:
            random_move = False
            move = moves_to_check[argmax][0]
            move_value = max_value

        critic_actor_value = self.net.forward(torch.cat([torch.from_numpy(state).to(device).float(), actor_action]))

        return move, [move_value, max_value, actor_action, critic_actor_value], random_move

    def off_policy_step(self, state, move_ind, info=None):
        """
        Selects information to save to memory when leaning off-policy.
        """

        move, info_, random = self.on_policy_step(state, info)
        move_value = self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[move_ind])]).to(device).float())

        return [move_value, info_[1], info_[2], info_[3]]

    def learn(self):
        """
        Calculates loss and runs backpropagation.
        """

        p1_loss = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves)
        p2_loss = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves)
        loss = p1_loss + p2_loss
        self.optimizer.zero_grad()
        loss.backward()

        #wolp_loss = 0.
        #for game in self.memory_1.game_log:
        #    wolp_loss -= torch.sum(torch.cat(game[-1][3]))
        #for game in self.memory_2.game_log:
        #    wolp_loss = -torch.sum(torch.cat(game[-1][3]))
        #torch.autograd.set_detect_anomaly(True)
        #self.actor_optimizer.zero_grad()
        #wolp_loss.backward()

        self.optimizer.step()
        #self.actor_optimizer.step()

        loss = loss #+ wolp_loss
        return float(loss.detach().cpu().numpy())

    def save(self, name, info=None):
        """
        Saves network parameters to memory.
        """

        j = info[0]
        torch.save(self.net.state_dict(), './saves/{}'.format(name + str(j)))
