import numpy as np
import torch
import torch.nn as nn
from models.neural_network import NN
from templates.agent import QuoridoorAgent

from parameters import Parameters
parameters = Parameters()

# parameters
actor_num_hidden = parameters.actor_num_hidden
actor_size_hidden = parameters.actor_size_hidden
input_dim = parameters.bot_in_dimension
actor_output_dim = parameters.bot_out_dimension
softmax_regularizer = parameters.softmax_regularizer


class Actor(NN):
    """
    Actor class for trainng using an actor-critic algorithm (to be used with a
    Q-network as a critic).

    save_name: name of the save file to be loaded.
    """

    def __init__(self, save_name=None):
        self.input_size = input_dim
        self.actor_output_dim = actor_output_dim
        super().__init__(self.input_size, actor_size_hidden, actor_num_hidden, actor_output_dim)
        self.softmax = nn.Softmax(dim=0)

        if save_name is not None:
            self.load_state_dict(torch.load('./saves/' + save_name + 'ACTOR'))

    def forward(self, x):
        output = self.feed_forward(x)
        return softmax_regularizer * self.softmax(output) + (1 - softmax_regularizer) / self.actor_output_dim

    def move(self, x):
        probabilities = self.forward(x)
        choice = np.random.choice(len(probabilities), p=probabilities.cpu().detach().numpy())
        move = np.zeros(len(probabilities))
        move[choice] = 1
        probability = probabilities[choice]
        return move, probabilities, probability


class ActorBot(QuoridoorAgent):

    def __init__(self, save_name, good=False):
        """
        Agent for running the Q network class in testing.

        save_name: name of the save from the trainer.
        """

        super().__init__()
        self.net = Actor()
        if not good:
            self.net.load_state_dict(torch.load('./saves/{}'.format(save_name), map_location=torch.device('cpu')))
        if good:
            self.net.load_state_dict(torch.load('./good_saves/{}'.format(save_name), map_location=torch.device('cpu')))
        self.possible_moves = [np.zeros(actor_output_dim) for _ in range(actor_output_dim)]
        for i in range(len(self.possible_moves)):
            self.possible_moves[i][i] = 1

    def move(self, state):
        return self.net.move(torch.from_numpy(state).float())[0]
