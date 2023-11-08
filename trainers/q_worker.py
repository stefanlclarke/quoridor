import torch
from parameters import Parameters
from loss_functions.sarsa_loss import sarsa_loss
from trainers.qtrainer import QTrainer
from models.q_models import QNetConv
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


class QWorker(mp.Process, QTrainer):
    def __init__(self, global_optimizer, res_queue, global_qnet, iterations=1, worker_it=1,
                 games_per_worker=games_per_iter,
                 stat_storage=None, net=None, convolutional=False):
        """
        Handles the training of an actor and a Q-network using an actor
        critic algorithm. Used in multiprocessing.
        """
        if convolutional:
            net = QNetConv()

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

    def push(self):

        """
        Calculates loss and does backpropagation.
        """

        critic_p1_loss, advantage_1 = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves, printing=False,
                                                 return_advantage=True)
        critic_p2_loss, advantage_2 = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves, printing=False,
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
        for i in range(self.iterations):
            for j in range(self.n_games_per_worker):
                self.play_game(info=[self.worker_it])
                self.stat_storage.n_games_played += 1
                self.log_memories()
            loss = self.push()
            self.net.pull(self.global_net)
            self.reset_memories()
            self.res_queue.put([i * j, loss])
        self.res_queue.put([None, loss])
