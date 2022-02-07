import numpy as np
import torch
import torch.nn as nn
import copy
from models.neural_network import NN

from parameters import Parameters
parameters = Parameters()

#parameters
actor_num_hidden = parameters.actor_num_hidden
actor_size_hidden = parameters.actor_size_hidden
input_dim = parameters.bot_in_dimension
actor_output_dim = parameters.bot_out_dimension

class Actor(NN):
    def __init__(self, save_name = None):
        self.input_size = input_dim
        super().__init__(self.input_size, actor_size_hidden, actor_num_hidden, actor_output_dim)
        self.softmax = nn.Softmax(dim=0)

        if save_name is not None:
            self.load_state_dict(torch.load(name + 'ACTOR'))

    def forward(self, x):
        output = self.feed_forward(x)
        return self.softmax(output)

    def move(self, x):
        probabilities = self.forward(x)
        choice = np.random.choice(len(probabilities), p=probabilities.cpu().detach().numpy())
        move = np.zeros(len(probabilities))
        move[choice] = 1
        probability = probabilities[choice]
        return move, probabilities, probability
