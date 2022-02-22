import numpy as np
import torch
from parameters import Parameters
from models.q_models import QNet
import torch.nn as nn
from game.move_reformatter import move_reformatter, unformatted_move_to_index
import copy
import torch.optim as optim
from game.game_helper_functions import *
from loss_functions.sarsa_loss import sarsa_loss
from game.shortest_path import *
from loss_functions.ppo_loss import actor_loss
from trainers.qtrainer import QTrainer
import torch.multiprocessing as mp

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


class QWorker(mp.Process, QTrainer):
    def __init__(self, global_optimizer, res_queue, global_qnet, iterations=1, worker_it=1):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm. Used in multiprocessing.
        """
        mp.Process.__init__(self)
        QTrainer.__init__(self)

        self.global_net = global_qnet.to(device)

        self.optimizer = global_optimizer

        self.learning_iterations_so_far = 0
        self.net.pull(self.global_net)

        self.res_queue = res_queue
        self.iterations = iterations
        self.worker_it = worker_it

    def push(self):

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves, printing=False, return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves, printing=False, return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        self.optimizer.zero_grad()
        critic_loss.backward()

        for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

        loss = critic_loss
        self.learning_iterations_so_far += 1

        return float(loss.detach().cpu().numpy())

    def run(self):
        for i in range(self.iterations):
            self.play_game(info=[self.worker_it])
            self.log_memories()
            self.push()
            self.net.pull(self.global_net)
            self.reset_memories()
            self.res_queue.put(i)
        self.res_queue.put(None)
