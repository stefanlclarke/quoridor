import numpy as np
import torch
from matplotlib import pyplot as plt
import csv
import pandas as pd
from optimizers.shared_adam import SharedAdam
import torch.multiprocessing as mp
from trainers.ac_trainer import ACWorker
from models.actor_models import Actor
from models.critic_models import Critic


class ParallelACTrainer:

    def __init__(self, board_size, start_walls, critic_info, actor_info, decrease_epsilon_every=100,
                 games_per_iter=100, lambd=0.9, gamma=0.9, random_proportion=0.4,
                 qnet=None, actor=None, iterations_only_actor_train=0, convolutional=False, learning_rate=1e-4,
                 epsilon_decay=0.95, epsilon=0.4, minimum_epsilon=0.05, entropy_constant=1, max_grad_norm=1e5,
                 move_prob=0.4, minimum_move_prob=0.2, entropy_bias=0, save_name='', total_reset_every=np.inf,
                 central_actor=None, central_critic=None, cores=1, iterations_per_worker=100, n_workers=1,
                 save_every=100, save_folder=''):

        if central_actor is None:
            self.central_actor = Actor(actor_info['actor_num_hidden'], actor_info['actor_size_hidden'],
                                       actor_info['input_dim'], actor_info['actor_output_dim'],
                                       actor_info['softmax_regularizer'])
        else:
            self.central_actor = central_actor

        if central_critic is None:
            self.central_critic = Critic(critic_info['input_dim'], critic_info['critic_size_hidden'],
                                         critic_info['critic_num_hidden'])
        else:
            self.central_critic = central_critic

        self.global_optimizer = SharedAdam(self.central_critic.parameters())
        self.global_actor_opt = SharedAdam(self.central_actor.parameters())
        self.res_queue = mp.Queue()

        self.iterations_per_worker = iterations_per_worker
        self.board_size = board_size
        self.start_walls = start_walls
        self.critic_info = critic_info
        self.actor_info = actor_info
        self.games_per_iter = games_per_iter
        self.lambd = lambd
        self.gamma = gamma
        self.random_proportion = random_proportion
        self.qnet = qnet
        self.actor = actor
        self.iterations_only_actor_train = iterations_only_actor_train
        self.convolutional = convolutional
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.entropy_constant = entropy_constant
        self.max_grad_norm = max_grad_norm
        self.move_prob = move_prob
        self.minimum_move_prob = minimum_move_prob
        self.entropy_bias = entropy_bias
        self.cores = cores
        self.save_every = save_every

        trainers = [ACWorker(iterations_per_worker,
                             board_size,
                             start_walls,
                             critic_info,
                             actor_info,
                             1,
                             games_per_iter,
                             lambd,
                             gamma,
                             random_proportion,
                             qnet,
                             actor,
                             iterations_only_actor_train,
                             convolutional,
                             learning_rate,
                             epsilon_decay,
                             epsilon,
                             minimum_epsilon,
                             entropy_constant,
                             max_grad_norm,
                             move_prob,
                             minimum_move_prob,
                             entropy_bias,
                             save_name,
                             np.inf,
                             self.central_actor,
                             self.central_critic,
                             cores,
                             self.res_queue) for _ in range(n_workers)]
        self.trainers = trainers
        self.decrease_epsilon_every = decrease_epsilon_every
        self.total_reset_every = total_reset_every
        self.save_name = save_name
        self.save_folder = save_folder
        self.n_workers = n_workers

        if self.total_reset_every is None:
            self.total_reset_every = np.inf

    def train(self, number_iterations, print_every=10):

        i_sub = 0
        print_iteration('epoch', 'move legality', 'average reward', 'game len', 'off pol %', 'loss')
        self.put_in_csv(['epoch', 'move legality', 'average reward', 'game len', 'off pol %', 'loss'])


        # loop over iterations
        for i in range(number_iterations):

            if i % self.total_reset_every == 0:
                i_sub = i

            for w in self.trainers:
                w.j = (i - i_sub) // self.decrease_epsilon_every

            # start all workers
            [w.start() for w in self.trainers]

            # join the workers
            [w.join() for w in self.trainers]

            self.res_queue.put(None)

            # self.global_optimizer.step()
            # self.global_optimizer.zero_grad()

            # self.global_actor_opt.step()
            # self.global_actor_opt.zero_grad()

            central_state = self.central_actor.state_dict()
            for key in self.central_actor.state_dict():
                central_state[key] = central_state[key] * 0.
                for w in self.trainers:
                    central_state[key] += w.actor.state_dict()[key] / len(self.trainers)
            self.central_actor.load_state_dict(central_state)

            central_state = self.central_critic.state_dict()
            for key in self.central_critic.state_dict():
                central_state[key] = central_state[key] * 0.
                for w in self.trainers:
                    central_state[key] += w.net.state_dict()[key] / len(self.trainers)
            self.central_critic.load_state_dict(central_state)

            out_info = list(iter(self.res_queue.get, None))
            average_out = sum([np.array(o) for o in out_info]) / len(out_info)
            average_out[0] = i * self.iterations_per_worker

            if i % print_every == 0:
                print_iteration(*list(average_out))
                self.put_in_csv(list(average_out))
                self.recreate_csv_plot()

            # terminate the workers
            [w.terminate() for w in self.trainers]
            self.res_queue = mp.Queue()
            self.reset_workers()

            # save
            if i % self.save_every == 0:
                self.save(i)

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

        trainers = [ACWorker(self.iterations_per_worker,
                             self.board_size,
                             self.start_walls,
                             self.critic_info,
                             self.actor_info,
                             1,
                             self.games_per_iter,
                             self.lambd,
                             self.gamma,
                             self.random_proportion,
                             self.qnet,
                             self.actor,
                             self.iterations_only_actor_train,
                             self.convolutional,
                             self.learning_rate,
                             self.epsilon_decay,
                             self.epsilon,
                             self.minimum_epsilon,
                             self.entropy_constant,
                             self.max_grad_norm,
                             self.move_prob,
                             self.minimum_move_prob,
                             self.entropy_bias,
                             self.save_name,
                             np.inf,
                             self.central_actor,
                             self.central_critic,
                             self.cores,
                             self.res_queue) for _ in range(self.n_workers)]
        self.trainers = trainers


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)
