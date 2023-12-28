import numpy as np
import os
import torch
from matplotlib import pyplot as plt
import csv
import pandas as pd
from optimizers.shared_adam import SharedAdam
import torch.multiprocessing as mp
from trainers.ac_trainer import ACWorker
from models.actor_models import Actor
from models.critic_models import Critic, CriticConv
from config import config

MAX_SAVES = 500


class ParallelACTrainer:

    def __init__(self, critic_info, actor_info, save_name='', save_directory='',
                 central_actor=None, central_critic=None):

        if central_actor is None:
            self.central_actor = Actor(actor_info['actor_num_hidden'], actor_info['actor_size_hidden'],
                                       actor_info['input_dim'], actor_info['actor_output_dim'],
                                       actor_info['softmax_regularizer'])
        else:
            self.central_actor = central_actor

        if central_critic is None:
            if not config.USE_CONV_NET:
                self.central_critic = Critic(critic_info['input_dim'], critic_info['critic_size_hidden'],
                                             critic_info['critic_num_hidden'])
            else:
                self.central_critic = CriticConv(**critic_info)
        else:
            self.central_critic = central_critic

        self.global_optimizer = SharedAdam(self.central_critic.parameters())
        self.global_actor_opt = SharedAdam(self.central_actor.parameters())
        self.res_queue = mp.Queue()

        self.critic_info = critic_info
        self.actor_info = actor_info

        trainers = [ACWorker(critic_info=critic_info, actor_info=actor_info, res_q=self.res_queue, save_name=save_name,
                             save_directory=save_directory,
                             central_critic=self.central_critic, central_actor=self.central_actor)
                    for _ in range(config.N_CORES)]

        self.trainers = trainers
        self.save_name = save_name
        self.save_folder = save_directory
        self.max_saves = MAX_SAVES

        if config.TOTAL_RESET_EVERY is None or "None":
            self.total_reset_every = np.inf

    def train(self, number_iterations, print_every=10):

        i_sub = 0
        print_iteration('epoch', 'move legality', 'average reward', 'game len', 'off pol %', 'loss')

        df = pd.read_csv('{}.csv'.format(self.save_folder + '/' + self.save_name), index_col='epoch')
        if len(df) == 0:
            self.put_in_csv(['epoch', 'move legality', 'average reward', 'game len', 'off pol %', 'loss'])
            start = 0
        else:
            start = int(df.index[-1] + 1)

        # loop over iterations
        for i in range(start, number_iterations):
            if i % self.total_reset_every == 0:
                i_sub = i

            for w in self.trainers:
                w.j = (i - i_sub) // config.DECREASE_EPSILON_EVERY

            # start all workers
            [w.start() for w in self.trainers]

            # join the workers
            [w.join() for w in self.trainers]

            self.res_queue.put(None)

            out_info = list(iter(self.res_queue.get, None))
            average_out = sum([np.array(o) for o in out_info]) / len(out_info)
            average_out[0] = i * config.WORKER_GAMES_BETWEEN_TRAINS

            if i % print_every == 0:
                print_iteration(*list(average_out))
                self.put_in_csv(list(average_out))
                self.recreate_csv_plot()

            # terminate the workers
            [w.terminate() for w in self.trainers]
            self.res_queue = mp.Queue()
            self.reset_workers()

            # save
            if i % config.SAVE_EVERY == 0:
                self.save(i)
                self.purge()

    def save(self, j):
        """
        Saves network parameters to memory.
        """
        torch.save(self.central_critic.state_dict(), self.save_folder + '/saves/' + self.save_name + str(j))
        torch.save(self.central_actor.state_dict(), self.save_folder + '/saves/' + self.save_name + str(j) + 'ACTOR')

    def put_in_csv(self, info):

        with open('{}.csv'.format(self.save_folder + '/' + self.save_name), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(info)

    def recreate_csv_plot(self):
        df = pd.read_csv('{}.csv'.format(self.save_folder + '/' + self.save_name), index_col='epoch')
        categories = df.columns

        fig, ax = plt.subplots(len(categories), figsize=(20, 10))

        for i in range(len(categories)):
            category = categories[i]
            df[category].plot(ax=ax[i], label=category)
            ax[i].set_ylabel(category)
        plt.tight_layout()
        plt.savefig(self.save_folder + '/' + self.save_name + '_plot')
        plt.close()
        del df

    def reset_workers(self):

        trainers = [ACWorker(critic_info=self.critic_info, actor_info=self.actor_info, res_q=self.res_queue,
                             save_name=self.save_name,
                             save_directory=self.save_folder,
                             central_critic=self.central_critic, central_actor=self.central_actor)
                    for _ in range(config.N_CORES)]

        self.trainers = trainers

    def purge(self):
        old_models = os.listdir(self.save_folder + '/saves')

        if len(old_models) < self.max_saves * 2 + 1:
            return

        old_actors = [x for x in old_models if x[-5:] == 'ACTOR']
        prev_savenums = sorted([int(x[4:-5]) for x in old_actors])
        to_delete = prev_savenums[: - self.max_saves]

        for choice in to_delete:
            os.remove(self.save_folder + '/saves' + '/' + self.save_name + str(choice) + 'ACTOR')
            os.remove(self.save_folder + '/saves' + '/' + self.save_name + str(choice))


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)
