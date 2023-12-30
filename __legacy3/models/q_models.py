import numpy as np
import torch
from models.neural_network import ConvNN, NN
from templates.agent import QuoridoorAgent


class QNetConv(ConvNN):

    def __init__(self, input_dim, actor_output_dim, conv_sidelen, num_conv, conv_internal_channels, linear_in_dim,
                 hidden_size, kernel_size):
        """
        Q network class for learning via Sarsa Lambda with convolutional layers.
        """

        self.input_size = input_dim + actor_output_dim
        self.conv_sidelen = conv_sidelen
        self.conv_in_channels = 4
        self.num_conv = num_conv
        self.conv_internal_channels = conv_internal_channels
        self.linear_in_dim = linear_in_dim
        super().__init__(self.conv_sidelen, self.conv_in_channels,
                         self.conv_internal_channels, self.linear_in_dim + actor_output_dim,
                         hidden_size, self.num_conv, hidden_size, 1, kernel_size)

    def forward(self, x):
        return self.feed_forward(x)


class QNet(NN):

    def __init__(self, input_dim, actor_output_dim, hidden_size, num_hidden):
        """
        Q network class for learning via Sarsa Lambda.
        """

        self.input_size = input_dim + actor_output_dim
        super().__init__(self.input_size, hidden_size, num_hidden, 1)

    def forward(self, x):
        return self.feed_forward(x)


class QNetBot(QuoridoorAgent):

    def __init__(self, save_name, nn_parameters, convolutional=False):
        """
        Agent for running the Q network class in testing.

        save_name: name of the save from the trainer.
        """

        super().__init__()
        if not convolutional:
            self.net = QNet(**nn_parameters)
        else:
            self.net = QNetConv(**nn_parameters)

        self.net.load_state_dict(torch.load(save_name, map_location=torch.device('cpu')))

        self.possible_moves = [np.zeros(nn_parameters['actor_output_dim'])
                               for _ in range(nn_parameters['actor_output_dim'])]
        for i in range(len(self.possible_moves)):
            self.possible_moves[i][i] = 1
        self.nn_parameters = nn_parameters

    def move(self, state):
        move_actions = [np.concatenate([state, move]) for move in self.possible_moves]
        values = np.concatenate([self.net.forward(torch.from_numpy(action).float()).detach().numpy()
                                 for action in move_actions])
        best_index = np.argmax(values)
        print(values)
        print(best_index)
        move = np.zeros(self.nn_parameters['actor_output_dim'])
        move[best_index] = 1
        return move
