from parameters import Parameters
from optimizers.shared_adam import SharedAdam
from loss_functions.sarsa_loss_simplified import sarsa_loss
from __legacy2.q_worker import QWorker
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import torch

try:
    set_start_method('spawn')
except RuntimeError:
    pass

parameters = Parameters()
random_proportion = parameters.random_proportion
games_per_worker = parameters.games_between_backprops
iters_per_worker = parameters.backprops_per_worker
total_reset_every = parameters.total_reset_every
learning_rate = parameters.learning_rate


class ParallelTrainer:
    def __init__(self, number_workers, global_critic, save_name, iterations_per_worker=1, save_freq=1,
                 convolutional=False):
        """
        Template class for training AI on Quoridor.
        """
        self.critic = global_critic
        self.iterations_per_worker = iterations_per_worker
        self.optimizer = SharedAdam(self.critic.parameters(), lr=learning_rate)
        self.number_workers = number_workers
        self.save_name = save_name
        self.save_freq = save_freq
        self.games_played = 0
        self.stats = TrainingStatistics()
        self.workers = []
        self.convolutional = convolutional
        self.total_reset_every = total_reset_every
        self.reset_workers()

    def reset_workers(self, worker_it=1):
        self.res_queue = mp.Queue()
        self.workers = [QWorker(self.optimizer, self.res_queue, self.critic, iterations=self.iterations_per_worker,
                                worker_it=worker_it, stat_storage=self.stats, convolutional=self.convolutional,
                                games_per_worker=games_per_worker)
                        for _ in range(self.number_workers)]

    def train(self, number_iterations):

        # for tracking results
        print_iteration('epoch', 'loss', 'n games', 'move_legality', 'average reward', 'average game len')
        n_games_played = 0
        n_sample = len(self.workers)
        i_normalizer = 0

        # loop over iterations
        for i in range(number_iterations):

            # start all workers
            [w.start() for w in self.workers]

            # join the workers
            [w.join() for w in self.workers]

            # terminate the workers
            [w.terminate() for w in self.workers]

            # get loss
            self.res_queue.put(None)
            out_info = list(iter(self.res_queue.get, None))
            losses = [o[0] for o in out_info]
            game_info = [o[1] for o in out_info]
            game_info = sum(game_info) / len(game_info)

            [w.reset_memories() for w in self.workers]

            # save a game-play if necessary
            if i % parameters.save_game_every == 0:
                self.workers[0].reset_memories()
                self.workers[0].play_game(info=[i - i_normalizer], printing=True, random_start=False)
                self.workers[0].log_memories()
                critic_p1_loss, advantage_1 = sarsa_loss(self.workers[0].memory_1, self.workers[0].net, 1,
                                                         self.workers[0].possible_moves, printing=True,
                                                         return_advantage=True)
                critic_p2_loss, advantage_2 = sarsa_loss(self.workers[0].memory_2, self.workers[0].net, 1,
                                                         self.workers[0].possible_moves, printing=True,
                                                         return_advantage=True)
                self.workers[0].save_most_recent_play(f'play{i}')
                print_iteration('epoch', 'loss', 'n games', 'move_legality', 'average reward', 'average game len')

            # reset the workers
            self.reset_workers(worker_it=i - i_normalizer)

            # get loss and print
            avg_loss = sum(losses) / n_sample
            n_games_played += n_sample * games_per_worker * iters_per_worker
            print_iteration(i, avg_loss, n_games_played, game_info[-4], game_info[-3], game_info[-2])

            # save if we reach saving iteration
            if i % self.save_freq == 0:
                self.save(self.save_name, i)

            if self.total_reset_every is not None:
                if i % self.total_reset_every == 0:
                    i_normalizer = i

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


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)