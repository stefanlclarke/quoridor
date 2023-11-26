import torch
from parameters import Parameters
from loss_functions.sarsa_loss import sarsa_loss
from __legacy2.ppo_loss import actor_loss
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
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion
entropy_constant = parameters.entropy_constant
max_grad_norm = parameters.max_grad_norm


class ACWorker(mp.Process, ACTrainer):
    def __init__(self, global_optimizer, global_actor_optimizer, res_queue, global_qnet, global_actor,
                 iterations_only_actor_train=0, iterations=1):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm. Used in multiprocessing.
        """

        mp.Process.__init__(self)
        ACTrainer.__init__(self)

        self.global_net = global_qnet.to(device)
        self.global_actor = global_actor.to(device)

        self.optimizer = global_optimizer
        self.actor_opt = global_actor_optimizer

        self.learning_iterations_so_far = 0
        self.iterations_only_actor_train = iterations_only_actor_train

        self.actor.pull(self.global_actor)
        self.net.pull(self.global_net)

        self.res_queue = res_queue
        self.iterations = iterations

    def push(self):
        if self.learning_iterations_so_far >= self.iterations_only_actor_train:
            train_critic = True
        else:
            train_critic = False

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves, printing=False,
                                                 return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves, printing=False,
                                                 return_advantage=True)
        critic_loss = critic_p1_loss + critic_p2_loss

        actor_p1_loss = actor_loss(self.memory_1, advantage_1, entropy_constant=entropy_constant)
        actor_p2_loss = actor_loss(self.memory_2, advantage_2, entropy_constant=entropy_constant)
        actor_loss_val = actor_p1_loss + actor_p2_loss

        self.optimizer.zero_grad()
        self.actor_opt.zero_grad()

        if train_critic:
            critic_loss.backward()

        actor_loss_val.backward()

        for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        for lp, gp in zip(self.actor.parameters(), self.global_actor.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()
        self.actor_opt.step()

        loss = actor_loss_val + critic_loss
        self.learning_iterations_so_far += 1

        return float(loss.detach().cpu().numpy())

    def run(self):
        for _ in range(self.iterations):
            self.play_game()
            print('completed a game')
            self.log_memories()
            self.push()
            self.actor.pull(self.global_actor)
            self.net.pull(self.global_net)
            self.reset_memories()
            self.res_queue.put(None)
