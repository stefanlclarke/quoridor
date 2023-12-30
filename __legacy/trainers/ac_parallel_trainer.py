from game.game import Quoridor
from parameters import Parameters
import numpy as np
from models.memory import Memory
from game.game.move_reformatter import *
from game.game.shortest_path import ShortestPathBot
import time
from optimizers.shared_adam import SharedAdam
from __legacy2.ac_worker import ACWorker
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

parameters = Parameters()
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion


class ParallelTrainer:
    def __init__(self, number_workers, global_critic, global_actor, iterations_per_worker=1):
        """
        Template class for training AI on Quoridor.
        """
        self.critic = global_critic
        self.actor = global_actor
        self.optimizer = SharedAdam(self.critic.parameters())
        self.actor_opt = SharedAdam(self.actor.parameters())
        self.res_queue = mp.Queue()
        self.workers = [ACWorker(self.optimizer, self.actor_opt, self.res_queue, self.critic, self.actor,
                                 iterations=iterations_per_worker) for _ in range(number_workers)]

    def train(self, number_iterations):
        for _ in range(number_iterations):
            [w.start() for w in self.workers]
            res = []
            while True:
                r = self.res_queue.get()
                if r is None:
                    break
            [w.join() for w in self.workers]
            [w.terminate() for w in self.workers]
