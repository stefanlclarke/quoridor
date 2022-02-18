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

possible_moves = [np.array(move) for move in parameters.possible_moves]

class WolpActor(NN):
    """
    Actor class for trainng using a Wolpertinger actor-critic algorithm (to be used with a
    Q-network as a critic).

    save_name: name of the save file to be loaded.
    """

    def __init__(self, save_name = None):
        self.input_size = input_dim
        super().__init__(self.input_size, actor_size_hidden, actor_num_hidden, actor_output_dim)

        if save_name is not None:
            self.load_state_dict(torch.load('./saves/' + save_name + 'ACTOR'))

    def forward(self, x):
        output = self.feed_forward(x)
        return output

def k_nearest(actor_out, k):
    act_out = actor_out.cpu().detach().numpy()
    distances = [(move, act_out - move) for move in possible_moves]
    distances.sort(key = lambda x: np.linalg.norm(x[1]))
    return distances[:k]
