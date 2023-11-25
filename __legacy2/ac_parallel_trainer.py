import torch
from parameters import Parameters
from optimizers.shared_adam import SharedAdam
from __legacy2.ac_worker import ACWorker, WeightClipper
import torch.multiprocessing as mp
from loss_functions.sarsa_loss_ac import sarsa_loss_ac
from loss_functions.actor_loss import actor_loss
import numpy as np

parameters = Parameters()
random_proportion = parameters.random_proportion
random_proportion = parameters.random_proportion
games_per_worker = parameters.games_between_backprops
iters_per_worker = parameters.backprops_per_worker
save_freq = parameters.save_every
total_reset_every = parameters.total_reset_every


class ParallelTrainer:
    def __init__(self, number_workers, global_critic, global_actor, save_name='test', iterations_per_worker=1,
                 convolutional=False):
        """
        Template class for training AI on Quoridor.
        """
        self.critic = global_critic
        self.actor = global_actor
        self.optimizer = SharedAdam(self.critic.parameters())
        self.actor_opt = SharedAdam(self.actor.parameters())
        self.res_queue = mp.Queue()
        self.workers = [ACWorker(self.optimizer, self.actor_opt, self.res_queue, self.critic, self.actor,
                                 total_epochs=iterations_per_worker, convolutional=convolutional)
                        for _ in range(number_workers)]
        self.number_workers = number_workers
        self.iterations_per_worker = iterations_per_worker
        self.save_freq = save_freq
        self.save_name = save_name
        self.convolutional = convolutional
        self.total_reset_every = total_reset_every
        self.clipper = WeightClipper()

    def reset_workers(self, worker_it=1):
        self.res_queue = mp.Queue()
        self.workers = [ACWorker(self.optimizer, self.actor_opt, self.res_queue, self.critic, self.actor,
                                 total_epochs=self.iterations_per_worker, convolutional=self.convolutional,
                                 worker_it=worker_it)
                        for _ in range(self.number_workers)]

    def train(self, number_iterations):

        # for tracking results
        print_iteration('epoch', 'loss', 'actor loss', 'entropy loss', 'critic loss',
                        'n games', 'move_legality',
                        'average reward', 'average game len')
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
                print([np.where(x == 1) for x in self.workers[0].memory_1.game_log[0][0]])
                print([np.where(x == 1) for x in self.workers[0].memory_1.game_log[0][1]])
                print(self.workers[0].memory_1.game_log[0][2])
                critic_p1_loss, advantage_1 = sarsa_loss_ac(self.workers[0].memory_1, self.workers[0].net, 0,
                                                            self.workers[0].possible_moves, printing=True,
                                                            return_advantage=True)
                critic_p2_loss, advantage_2 = sarsa_loss_ac(self.workers[0].memory_2, self.workers[0].net, 0,
                                                            self.workers[0].possible_moves, printing=True,
                                                            return_advantage=True)
                self.workers[0].save_most_recent_play(f'play{i}')
                actor_p1_loss, entropy_p1_loss = actor_loss(self.workers[0].memory_1, advantage_1,
                                                            entropy_constant=parameters.entropy_constant)
                actor_p2_loss, entropy_p2_loss = actor_loss(self.workers[0].memory_2, advantage_2,
                                                            entropy_constant=parameters.entropy_constant)
                print_iteration('epoch', 'loss', 'actor loss', 'entropy loss', 'critic loss',
                                'n games', 'move_legality',
                                'average reward', 'average game len')

            # reset the workers
            self.reset_workers(worker_it=i - i_normalizer)

            # get loss and print
            avg_losses = sum(losses) / n_sample
            n_games_played += n_sample * games_per_worker * iters_per_worker
            print_iteration(i, avg_losses[0], avg_losses[1], avg_losses[2], avg_losses[3], n_games_played,
                            game_info[-4], game_info[-3], game_info[-2]
                            )

            # clip actor weights
            self.actor.apply(self.clipper)

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
        torch.save(self.critic.state_dict(), './saves/{}_critic'.format(name + str(j)))
        torch.save(self.actor.state_dict(), './saves/{}_actor'.format(name + str(j)))


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)
