import torch
import time
import numpy as np
from parameters import Parameters
from loss_functions.sarsa_loss_simplified import sarsa_loss
from trainers.qtrainer import QTrainer
from models.q_models import QNetConv, QNet
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
random_proportion = parameters.random_proportion
entropy_constant = parameters.entropy_constant
max_grad_norm = parameters.max_grad_norm
backwards_per_worker = parameters.backwards_per_worker


class QWorker(mp.Process, QTrainer):
    def __init__(self, global_optimizer, res_queue, global_qnet, iterations=1, worker_it=1,
                 games_per_worker=1,
                 stat_storage=None, net=None, convolutional=False):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm. Used in multiprocessing.
        """
        if convolutional:
            net = QNetConv()
        else:
            net = QNet()

        mp.Process.__init__(self)
        QTrainer.__init__(self, net=net)

        self.global_net = global_qnet.to(device)

        self.optimizer = global_optimizer

        self.learning_iterations_so_far = 0
        self.net.pull(self.global_net)

        self.res_queue = res_queue
        self.iterations = iterations
        self.worker_it = worker_it

        self.n_games_played = 0
        self.stat_storage = stat_storage
        self.n_games_per_worker = games_per_worker
        self.backwards_per_worker = backwards_per_worker

    def push(self, epoch=0):

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss(self.memory_1, self.net, epoch, self.possible_moves, printing=False,
                                                 return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss(self.memory_2, self.net, epoch, self.possible_moves, printing=False,
                                                 return_advantage=True)
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
        time_info_total = []
        for i in range(self.iterations):
            self.reset_memories()
            for j in range(self.n_games_per_worker):
                game_info = self.play_game(info=[self.worker_it])
                game_timing_array = np.array(list(game_info.values()) + [0])
                self.stat_storage.n_games_played += 1
                self.log_memories()
                time_info_total.append(game_timing_array)
            for j in range(self.backwards_per_worker):
                t0 = time.time()
                loss = self.push(epoch=1)
                self.net.pull(self.global_net)
                self.optimizer.zero_grad()
                t1 = time.time()
                backward_timing_array = np.array([0 for _ in range(len(game_timing_array) - 1)] + [t1 - t0])
                time_info_total.append(backward_timing_array)
        total_time = sum(time_info_total) / (self.iterations * self.n_games_per_worker)
        self.res_queue.put([loss, total_time])
