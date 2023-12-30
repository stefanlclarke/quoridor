from game.game import Quoridor
from parameters import Parameters
import numpy as np
from models.memory import Memory
from game.game.move_reformatter import *
from game.game.shortest_path import ShortestPathBot
import time
from optimizers.shared_adam import SharedAdam
from __legacy2.q_worker import QWorker
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import torch

try:
     set_start_method('spawn')
except RuntimeError:
    pass

parameters = Parameters()
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion

class ParallelTrainer:
    def __init__(self, number_workers, global_critic, save_name, iterations_per_worker=1, save_freq=1):
        """
        Template class for training AI on Quoridor.
        """
        self.critic = global_critic
        self.iterations_per_worker = iterations_per_worker
        self.optimizer = SharedAdam(self.critic.parameters())
        self.number_workers = number_workers
        self.save_name = save_name
        self.save_freq = save_freq
        self.reset_workers()

    def reset_workers(self, worker_it=1):
        self.res_queue = mp.Queue()
        self.workers = [QWorker(self.optimizer, self.res_queue, self.critic, iterations=self.iterations_per_worker, worker_it=worker_it) for _ in range(self.number_workers)]

    def train(self, number_iterations):
        for i in range(number_iterations):
            [w.start() for w in self.workers]
            res = []
            while True:
                r = self.res_queue.get()
                if r is None:
                    break
            [w.join() for w in self.workers]
            [w.terminate() for w in self.workers]
            self.reset_workers(worker_it=i)

            if i % self.save_freq == 0:
                self.save(self.save_name, i)

    def save(self, name, j):
        """
        Saves network parameters to memory.
        """
        torch.save(self.critic.state_dict(), './saves/{}'.format(name + str(j)))
