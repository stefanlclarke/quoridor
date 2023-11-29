import numpy as np
import torch
from templates.trainer import Trainer
from models.q_models import QNet
import torch.optim as optim
from loss_functions.sarsa_loss_simplified import sarsa_loss

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


class QTrainer(Trainer):
    def __init__(self, board_size, start_walls, decrease_epsilon_every, random_proportion, games_per_iter,
                 qnet_parameters, learning_rate, epsilon, minimum_epsilon, minimum_move_prob, lambd, gamma,
                 save_name, epsilon_decay, net=None, total_reset_every=np.inf):
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
                         save_name=save_name)

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

        # initialize optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

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

    def learn(self):
        """
        Calculates loss and runs backpropagation.
        """

        # get losses
        p1_loss = sarsa_loss(memory=self.memory_1, net=self.net, epoch=0, possible_moves=self.possible_moves,
                             lambd=self.lambd, gamma=self.gamma)
        p2_loss = sarsa_loss(self.memory_2, self.net, epoch=0, possible_moves=self.possible_moves,
                             lambd=self.lambd, gamma=self.gamma)

        # get total loss
        loss = p1_loss + p2_loss
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
        torch.save(self.net.state_dict(), self.save_name + str(j))
