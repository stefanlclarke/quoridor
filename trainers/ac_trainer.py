import numpy as np
import torch
from parameters import Parameters
from models.q_models import QNet
from models.actor_models import Actor
import torch.nn as nn
from game.move_reformatter import move_reformatter, unformatted_move_to_index
import copy
import torch.optim as optim
from game.game_helper_functions import *
from loss_functions.sarsa_loss import sarsa_loss
from game.shortest_path import *
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
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion
entropy_constant = parameters.entropy_constant
max_grad_norm = parameters.max_grad_norm


class ACTrainer(Trainer):
    def __init__(self, qnet=None, actor=None):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm.
        """

        super().__init__(number_other_info=4)
        if qnet is None:
            self.net = QNet().to(device)
        else:
            self.net = net.to(device)
        if actor is None:
            self.actor = Actor().to(device)
        else:
            self.actor = actor.to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def on_policy_step(self, state, info):
        """
        Handles game decisions and chooses information to save to memory.
        """

        if info is None:
            decay = 1.
        else:
            decay = info[0]

        state_torch = torch.from_numpy(state).to(device).float()
        actor_move, actor_probabilities, actor_probability = self.actor.move(state_torch)
        state_action = torch.cat([state_torch, torch.from_numpy(actor_move).to(device)])
        critic_action_value = self.net.feed_forward(state_action.float())

        move = actor_move
        probability = actor_probability
        random_move = False
        move_critic_action_value = critic_action_value
        return move, [critic_action_value, critic_action_value, probability, actor_probabilities], False

    def off_policy_step(self, state, move_ind, info):
        """
        Chooses information to save to memory when learning off-policy.
        """

        state_torch = torch.from_numpy(state).to(device).float()
        actor_move, actor_probabilities, actor_probability = self.actor.move(state_torch)
        state_action = torch.cat([state_torch, torch.from_numpy(actor_move).to(device)])
        critic_action_value = self.net.feed_forward(state_action.float())

        move = actor_move
        probability = actor_probability
        random_move = False

        probability = actor_probabilities[move_ind]
        state_action_new = torch.cat([state_torch, torch.from_numpy(move).to(device)])
        move_critic_action_value = self.net.feed_forward(state_action_new.float())

        return [move_critic_action_value, critic_action_value, probability, actor_probabilities]

    def save(self, name, info=None):
        """
        Saves network parameters to memory.
        """

        j = info[0]
        torch.save(self.net.state_dict(), './saves/{}'.format(name + str(j)))
        torch.save(self.actor.state_dict(), './saves/{}'.format(name + str(j) + 'ACTOR'))

    def learn(self, train_critic=True):
        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves, printing=False, return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves, printing=False, return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        actor_p1_loss = actor_loss(self.memory_1, advantage_1, entropy_constant=entropy_constant)
        actor_p2_loss = actor_loss(self.memory_2, advantage_2, entropy_constant=entropy_constant)
        actor_loss_val = actor_p1_loss + actor_p2_loss

        self.optimizer.zero_grad()
        self.actor_opt.zero_grad()

        if train_critic:
            critic_loss.backward()
            self.optimizer.step()

        actor_loss_val.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_opt.step()

        loss = actor_loss_val + critic_loss
        return float(loss.detach().cpu().numpy())
