import numpy as np
import torch
from templates.trainer import Trainer
from parameters import Parameters
from models.q_models import QNet
import torch.optim as optim
from loss_functions.sarsa_loss_simplified import sarsa_loss

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


class QTrainer(Trainer):
    def __init__(self, net=None):
        """
        Handles the training of a Q-network using Sarsa Lambda.
        """

        # initialize superclass
        super().__init__()

        # decide on type of neural network to use
        if net is None:
            self.net = QNet().to(device)
        else:
            self.net = net.to(device)

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
        for i in range(parameters.bot_out_dimension):
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
        if u < max([epsilon**decay, minimum_epsilon]):
            random_move = True
            if v < max([move_prob**decay, minimum_move_prob]):
                move = np.random.choice(4)
            else:
                move = np.random.choice(parameters.bot_out_dimension - 4) + 4
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
        for i in range(parameters.bot_out_dimension):
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
        p1_loss = sarsa_loss(self.memory_1, self.net, 0, self.possible_moves)
        p2_loss = sarsa_loss(self.memory_2, self.net, 0, self.possible_moves)

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
        torch.save(self.net.state_dict(), './saves/{}'.format(name + str(j)))
