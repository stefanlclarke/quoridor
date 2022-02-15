import numpy as np
import torch
from templates.trainer_parallel import ParallelTrainer
from parameters import Parameters
from models.q_models import QNet
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


class ParallelQTrainer(ParallelTrainer):
    def __init__(self, net=None):
        """
        Handles the training of a Q-network using Sarsa Lambda.
        """

        super().__init__()
        if net is None:
            self.net = QNet().to(device)
        else:
            self.net = net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def on_policy_step(self, state, info=None):
        """
        Determines on-policy action and selects information to choose to memory.
        An epsilon-greedy policy is used.
        """

        if info is None:
            decay = 1.
        else:
            decay = info[0]
        values = []
        for i in range(parameters.bot_out_dimension):
            values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[i])]).to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()
        print(values[-1].device)
        print(next(self.net.parameters()).device)
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())
        u = np.random.uniform()
        v = np.random.uniform()
        if u < max([epsilon**decay, minimum_epsilon]):
            random_move = True
            if v < max([move_prob**decay, minimum_move_prob]):
                move = np.random.choice(4)
            else:
                move = np.random.choice(parameters.bot_out_dimension - 4) + 4
        else:
            random_move = False
            move = argmax

        return self.possible_moves[move], [values[move], values[argmax]], random_move

    def off_policy_step(self, state, move_ind, info=None):
        """
        Selects information to save to memory when leaning off-policy.
        """

        values = []
        for i in range(parameters.bot_out_dimension):
            values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[i])]).to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())
        return [values[move_ind], values[argmax]]

    def learn(self):
        """
        Calculates loss and runs backpropagation.
        """

        p1_loss = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves)
        p2_loss = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves)
        loss = p1_loss + p2_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().numpy())

    def save(self, name, info=None):
        """
        Saves network parameters to memory.
        """

        j = info[0]
        torch.save(self.net.state_dict(), './saves/{}'.format(name + str(j)))
