import torch
import numpy as np
import time
from parameters import Parameters
from loss_functions.sarsa_loss_ac import sarsa_loss_ac
from loss_functions.actor_loss import actor_loss
from trainers.ac_trainer import ACTrainer
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
games_per_worker = parameters.games_between_backprops
backwards_per_worker = parameters.backwards_per_worker
train_actor = parameters.train_actor
train_critic = parameters.train_critic
n_iterations_only_critic = parameters.n_iterations_only_critic


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-parameters.actor_weight_clip, parameters.actor_weight_clip)


class ACWorker(mp.Process, ACTrainer):
    def __init__(self, global_optimizer, global_actor_optimizer, res_queue, global_qnet, global_actor,
                 iterations_only_actor_train=0, total_epochs=1, convolutional=False, worker_it=1):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm. Used in multiprocessing.
        """

        mp.Process.__init__(self)
        ACTrainer.__init__(self, convolutional=convolutional)
        self.clipper = WeightClipper()

        self.global_net = global_qnet.to(device)
        self.global_actor = global_actor.to(device)

        self.optimizer = global_optimizer
        self.actor_opt = global_actor_optimizer

        self.learning_iterations_so_far = 0
        self.iterations_only_actor_train = iterations_only_actor_train

        self.actor.pull(self.global_actor)
        self.net.pull(self.global_net)

        self.res_queue = res_queue
        self.iterations = total_epochs

        self.n_games_per_worker = games_per_worker
        self.backwards_per_worker = backwards_per_worker
        self.worker_it = worker_it

    def push(self, train_critic=True, train_actor=True, push_epoch=0):

        """
        Calculates loss and does backpropagation.
        """

        if self.worker_it < n_iterations_only_critic:
            train_actor = False

        critic_p1_loss, advantage_1 = sarsa_loss_ac(self.memory_1, self.net, push_epoch, self.possible_moves,
                                                    printing=False,
                                                    return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss_ac(self.memory_2, self.net, push_epoch, self.possible_moves,
                                                    printing=False,
                                                    return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        actor_p1_loss, entropy_p1_loss = actor_loss(self.memory_1, advantage_1,
                                                    entropy_constant=entropy_constant)
        actor_p2_loss, entropy_p2_loss = actor_loss(self.memory_2, advantage_2,
                                                    entropy_constant=entropy_constant)
        actor_loss_val = actor_p1_loss + actor_p2_loss + entropy_p2_loss + entropy_p1_loss

        self.optimizer.zero_grad()
        self.actor_opt.zero_grad()

        if train_critic:
            critic_loss.backward()
            for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
                gp._grad = lp.grad
            self.optimizer.step()
            self.net.apply(self.clipper)

        if train_actor:
            actor_loss_val.backward()
            for lp, gp in zip(self.actor.parameters(), self.global_actor.parameters()):
                gp._grad = lp.grad
            self.actor_opt.step()
            self.actor.apply(self.clipper)

        loss = actor_loss_val + critic_loss
        self.learning_iterations_so_far += 1

        return float(loss.detach().cpu().numpy()), float((actor_p1_loss + actor_p2_loss).detach().numpy()), \
            float((entropy_p1_loss + entropy_p2_loss).detach().numpy()), \
            float((critic_loss).detach().numpy())

    def run(self):
        time_info_total = []
        for i in range(self.iterations):
            for j in range(self.n_games_per_worker):
                game_info = self.play_game(info=[self.worker_it])
                game_timing_array = np.array(list(game_info.values()) + [0])
                self.log_memories()
                time_info_total.append(game_timing_array)
            for j in range(self.backwards_per_worker):
                t0 = time.time()
                loss, actor_loss, entropy_loss, critic_loss = self.push(train_actor=train_actor,
                                                                        train_critic=train_critic)
                loss_info = np.array([loss, actor_loss, entropy_loss, critic_loss])
                self.actor.pull(self.global_actor)
                self.net.pull(self.global_net)
                self.optimizer.zero_grad()
                t1 = time.time()
                backward_timing_array = np.array([0 for _ in range(len(game_timing_array) - 1)] + [t1 - t0])
                time_info_total.append(backward_timing_array)
            self.reset_memories()
        total_time = sum(time_info_total) / (self.iterations * self.n_games_per_worker)
        self.res_queue.put([loss_info, total_time])
