import numpy as np
import torch
from models.neural_network import ConvNN, NN
from templates.agent import QuoridoorAgent

from parameters import Parameters
parameters = Parameters()

critic_num_hidden = parameters.critic_num_hidden
critic_size_hidden = parameters.critic_size_hidden
input_dim = parameters.bot_in_dimension
actor_output_dim = parameters.bot_out_dimension
sidelen = parameters.sidelen
conv_internal_channels = parameters.conv_internal_channels
linear_in_dim = 2 * parameters.number_of_walls + 2
num_conv = parameters.num_conv
convolutional = parameters.convolutional
kernel_size = parameters.conv_kernel_size


class QNetConv(ConvNN):

    def __init__(self):
        """
        Q network class for learning via Sarsa Lambda with convolutional layers.
        """

        self.input_size = input_dim + actor_output_dim
        self.conv_sidelen = sidelen
        self.conv_in_channels = 4
        self.num_conv = num_conv
        self.conv_internal_channels = conv_internal_channels
        self.linear_in_dim = linear_in_dim
        super().__init__(self.conv_sidelen, self.conv_in_channels,
                         self.conv_internal_channels, self.linear_in_dim + actor_output_dim,
                         critic_size_hidden, self.num_conv, critic_num_hidden, 1, kernel_size)

    def forward(self, x):
        return self.feed_forward(x)


class QNet(NN):

    def __init__(self):
        """
        Q network class for learning via Sarsa Lambda.
        """

        self.input_size = input_dim + actor_output_dim
        super().__init__(self.input_size, critic_size_hidden, critic_num_hidden, 1)

    def forward(self, x):
        return self.feed_forward(x)


class QNetBot(QuoridoorAgent):

    def __init__(self, save_name, good=False):
        """
        Agent for running the Q network class in testing.

        save_name: name of the save from the trainer.
        """

        super().__init__()
        if not convolutional:
            self.net = QNet()
        else:
            self.net = QNetConv()
        if not good:
            self.net.load_state_dict(torch.load('./saves/{}'.format(save_name), map_location=torch.device('cpu')))
        if good:
            self.net.load_state_dict(torch.load('./good_saves/{}'.format(save_name), map_location=torch.device('cpu')))
        self.possible_moves = [np.zeros(actor_output_dim) for _ in range(actor_output_dim)]
        for i in range(len(self.possible_moves)):
            self.possible_moves[i][i] = 1

    def move(self, state):
        move_actions = [np.concatenate([state, move]) for move in self.possible_moves]
        values = np.concatenate([self.net.forward(torch.from_numpy(action).float()).detach().numpy()
                                 for action in move_actions])
        best_index = np.argmax(values)
        print(values)
        print(best_index)
        move = np.zeros(actor_output_dim)
        move[best_index] = 1
        return move
