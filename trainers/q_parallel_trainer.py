from parameters import Parameters
from optimizers.shared_adam import SharedAdam
from trainers.q_worker import QWorker
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import torch

try:
    set_start_method('spawn')
except RuntimeError:
    pass

parameters = Parameters()
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion


class ParallelTrainer:
    def __init__(self, number_workers, global_critic, save_name, iterations_per_worker=1, save_freq=1,
                 convolutional=False):
        """
        Template class for training AI on Quoridor.
        """
        self.critic = global_critic
        self.iterations_per_worker = iterations_per_worker
        self.optimizer = SharedAdam(self.critic.parameters())
        self.number_workers = number_workers
        self.save_name = save_name
        self.save_freq = save_freq
        self.games_played = 0
        self.stats = TrainingStatistics()
        self.workers = []
        self.convolutional = convolutional
        self.reset_workers()

    def reset_workers(self, worker_it=1):
        self.res_queue = mp.Queue()
        self.workers = [QWorker(self.optimizer, self.res_queue, self.critic, iterations=self.iterations_per_worker,
                                worker_it=worker_it, stat_storage=self.stats, convolutional=self.convolutional)
                        for _ in range(self.number_workers)]

    def train(self, number_iterations):

        # for tracking results
        print('{}\t\t{}\t\t{}'.format('epoch', 'loss', 'num games played'))
        n_games_played = 0

        # loop over iterations
        for i in range(number_iterations):

            # start all workers
            [w.start() for w in self.workers]

            # tracking loss and number of plays
            n_sample = 0
            avg_loss = 0
            while True:

                # get from the queue
                r = self.res_queue.get()
                avg_loss += r[1]
                n_sample += 1

                # if any process has finished stop
                if r[0] is None:
                    break

            # join the workers
            [w.join() for w in self.workers]

            # terminate the workers
            [w.terminate() for w in self.workers]

            # reset the workers
            self.reset_workers(worker_it=i)

            # get loss and print
            avg_loss = avg_loss / n_sample
            n_games_played += n_sample
            print_iteration(i, avg_loss, n_games_played)

            # save if we reach saving iteration
            if i % self.save_freq == 0:
                self.save(self.save_name, i)

    def save(self, name, j):
        """
        Saves network parameters to memory.
        """
        torch.save(self.critic.state_dict(), './saves/{}'.format(name + str(j)))


class TrainingStatistics:
    def __init__(self):
        """
        A class which stores statistics relating to the training run
        """

        self.n_games_played = 0


def print_iteration(epoch, loss, n_games):
    print('{:<5.0f}\t\t{:<5.5f}\t\t{:<5.0f}'.format(epoch, loss, n_games))