import numpy as np
from quoridor_env.config import game_config


class RandomAgent:
    def __init__(self):
        """
        Generic agent class.

        input_type: either 'board' or 'game'
        output_type: either 'one_hot' or 'true'
        """

        self.input_type = 'board'
        self.output_type = 'one_hot'
        self.output_dim = 4 + 2 * (game_config.BOARD_SIZE - 1)**2

    def move(self, input):
        """
        Should make the move for the agent.

        Returns a numpy array in the format dictated by the agent
        output type.
        """

        choice = np.random.choice(self.output_dim)
        vec = np.zeros(self.output_dim)
        vec[choice] = 1
        return vec
