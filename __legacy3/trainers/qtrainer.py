import numpy as np
import torch
from templates.trainer import Trainer
from models.q_models import QNet
import torch.optim as optim
from loss_functions.sarsa_loss_simplified import sarsa_loss
from config import config
import os

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

LOAD_FROM_LAST = 40


class QTrainer(Trainer):
    def __init__(self,
                 qnet_parameters,
                 save_name='',
                 net=None,
                 save_directory='',
                 total_reset_every=np.inf):
        
        board_size = config.BOARD_SIZE,
        start_walls = config.NUMBER_OF_WALLS,
        decrease_epsilon_every = config.DECREASE_EPSILON_EVERY,
        random_proportion = config.RANDOM_PROPORTION,
        games_per_iter = config.WORKER_GAMES_BETWEEN_TRAINS,
        learning_rate = config.Q_LEARNING_RATE,
        epsilon = config.EPSILON,
        minimum_epsilon = config.MINIMUM_EPSILON,
        minimum_move_prob = config.MINIMUM_MOVE_PROB,
        lambd = config.LAMBD,
        gamma = config.GAMMA,
        epsilon_decay = config.EPSILON_DECAY

        """
        Handles the training of a Q-network using Sarsa Lambda.
        """

        # initialize superclass
        super().__init__(board_size=board_size,
                         start_walls=start_walls,
                         number_other_info=2,
                         decrease_epsilon_every=decrease_epsilon_every,
                         random_proportion=random_proportion,
                         games_per_iter=games_per_iter,
                         total_reset_every=total_reset_every,
                         save_name=save_name,
                         save_directory=save_directory,
                         old_selfplay=config.OLD_SELFPLAY)

        # decide on type of neural network to use
        if net is None:
            self.net = QNet(**qnet_parameters).to(device)
        else:
            self.net = net.to(device)
        self.qnet_parameters = qnet_parameters
        self.bot_out_dimension = qnet_parameters['actor_output_dim']
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.minimum_move_prob = minimum_move_prob
        self.lambd = lambd
        self.gamma = gamma
        self.save_name = save_name
        self.epsilon_decay = epsilon_decay
        self.reload_distribution = config.RELOAD_DISTRIBUTION
        self.load_from_last = LOAD_FROM_LAST

        # initialize optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        if self.old_selfplay:
            self.loaded_net = QNet(**qnet_parameters).to(device)
            self.loaded_net.pull(self.net)

    def on_policy_step(self, state, info=None):
        """
        Determines on-policy action and selects information to choose to memory.
        An epsilon-greedy policy is used.

        inputs: 
            state: np.ndarray
                the game state

            info: list
                first element is the decay

        returns:
            move, [move value, best possible move value], bool (random move -> False)
        """

        # get the lr decay
        if info is None:
            decay = 1.
        else:
            decay = info[0]

        # get values of all potential moves
        values = []
        for i in range(self.bot_out_dimension):
            values.append(self.net.forward(torch.cat([torch.from_numpy(state),
                                                      torch.from_numpy(self.possible_moves[i])]).to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()

        # get best move
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())

        # decide whether to move randomly
        u = np.random.uniform()
        v = np.random.uniform()

        # figure out what the random move is
        if u < max([self.epsilon * self.epsilon_decay**decay, self.minimum_epsilon]):
            random_move = True
            if v < max([0, self.minimum_move_prob]):
                move = np.random.choice(4)
            else:
                move = np.random.choice(self.bot_out_dimension - 4) + 4
        else:
            random_move = False
            move = argmax

        # return the move
        return self.possible_moves[move], [values[move], values[argmax]], random_move

    def off_policy_step(self, state, move_ind, info=None):
        """
        Selects information to save to memory when leaning off-policy.
        """

        # work out the calues of possible moves
        values = []
        for i in range(self.bot_out_dimension):
            values.append(self.net.forward(torch.cat([torch.from_numpy(state),
                                                      torch.from_numpy(self.possible_moves[i])]).to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())

        # return value of move, value of best move
        return [values[move_ind], values[argmax]]

    def learn(self, side):
        """
        Calculates loss and runs backpropagation.
        """

        # get losses
        p1_loss = sarsa_loss(memory=self.memory_1, net=self.net, epoch=0, possible_moves=self.possible_moves,
                             lambd=self.lambd, gamma=self.gamma)
        p2_loss = sarsa_loss(self.memory_2, self.net, epoch=0, possible_moves=self.possible_moves,
                             lambd=self.lambd, gamma=self.gamma)

        # get total loss
        if side == 0:
            loss = p1_loss
        elif side == 1:
            loss = p2_loss
        self.optimizer.zero_grad()

        # backpropagate
        loss.backward()
        self.optimizer.step()

        # return the loss
        return float(loss.detach().cpu().numpy())

    def save(self, name, info=None):
        """
        Saves network parameters to memory.
        """

        j = info[0]
        torch.save(self.net.state_dict(), self.save_directory + '/' + self.save_name + str(j))

    def load_opponent(self, j=0):
        """
        Chooses an old version of self and loads it in as the opponent
        """

        old_models = os.listdir(self.save_directory + '/saves/')

        if len(old_models) == 0:
            self.net.pull(self.net)
            return

        old_nets = old_models
        prev_savenums = sorted([int(x[4:-5]) for x in old_nets])
        acceptable_choices = prev_savenums[-self.load_from_last:]

        if self.reload_distribution == "uniform":
            choice = np.random.choice(acceptable_choices)
        elif self.reload_distribution == "geometric":
            choice_ind = np.random.geometric(p=0.5)
            if choice_ind >= len(acceptable_choices):
                choice_ind = len(acceptable_choices)
            choice = acceptable_choices[-choice_ind]
        else:
            raise ValueError("invalid distribution")

        self.loaded_net.load_state_dict(torch.load(self.save_directory + '/saves/' + self.save_name + str(choice)))

    def loaded_on_policy_step(self, state, info=None):
        """
        Determines on-policy action and selects information to choose to memory.
        An epsilon-greedy policy is used.

        inputs: 
            state: np.ndarray
                the game state

            info: list
                first element is the decay

        returns:
            move, [move value, best possible move value], bool (random move -> False)
        """

        # get values of all potential moves
        values = []
        for i in range(self.bot_out_dimension):
            values.append(self.loaded_net.forward(torch.cat([torch.from_numpy(state),
                                                             torch.from_numpy(self.possible_moves[i])])
                                                  .to(device).float()))
        values_np = torch.cat(values).detach().cpu().numpy()

        # get best move
        argmax = np.argmax(values_np)
        argmax = np.random.choice(np.argwhere(values_np == values_np[argmax]).flatten())

        random_move = False
        move = argmax

        # return the move
        return self.possible_moves[move], [values[move], values[argmax]], random_move