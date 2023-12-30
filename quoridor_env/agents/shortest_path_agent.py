from quoridor_env.game.game import Quoridor
from quoridor_env.game.move_reformatter import unformatted_move_to_index
from quoridor_env.config import game_config
from quoridor_env.game.shortest_path import ShortestPathBot

import numpy as np


class ShortestPathAgent:
    def __init__(self, playing):
        """
        Generic agent class.

        input_type: either 'board' or 'game'
        output_type: either 'one_hot' or 'true'
        """

        self.input_type = 'board'
        self.output_type = 'one_hot'
        self.playing = playing
        self.game_clone = Quoridor(game_config.BOARD_SIZE, game_config.NUMBER_OF_WALLS)
        self.spbot = None

        if self.playing == 0:
            self.flip = False
        else:
            self.flip = True

    def move(self, input):
        """
        Should make the move for the agent.

        Returns a numpy array in the format dictated by the agent
        output type.
        """
        board_input = input[:-2 * (1 + game_config.NUMBER_OF_WALLS)]
        board = board_input.reshape((game_config.BOARD_SIZE, game_config.BOARD_SIZE, 4))

        p1_pos = np.where(board[:, :, 2] == 1)
        p2_pos = np.where(board[:, :, 3] == 1)
        self.output_dim = 4 + 2 * (game_config.BOARD_SIZE - 1)**2

        self.game_clone.copy_board(board, p1_pos, p2_pos, 0, 0)

        locs = np.where(self.game_clone.board[:, :, 0:2] == 1)
        locs = np.block([[l] for l in locs]).T
        locs_up = [l[0:2] for l in locs if l[2] == 0]
        locs_right = [l[0:2] for l in locs if l[2] == 1]

        edges = []

        for wall_pos_command in locs_up:
            vertices = [wall_pos_command]
            edges += [(tuple(vertex), tuple(vertex + np.array([1, 0]))) for vertex in vertices]

        for wall_pos_command in locs_right:
            vertices = [wall_pos_command]
            edges += [(tuple(vertex), tuple(vertex + np.array([0, 1]))) for vertex in vertices]

        self.game_clone.board_graph.reconfigure_paths(edges)
        self.spbot = ShortestPathBot(1, self.game_clone.board_graph)
        move = self.spbot.move(self.game_clone.board, self.game_clone.board_graph)

        oh_move_ind = unformatted_move_to_index(move)
        oh_move = np.zeros(self.output_dim)
        oh_move[oh_move_ind] = 1

        return oh_move
