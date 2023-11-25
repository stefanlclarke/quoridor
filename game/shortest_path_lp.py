import numpy as np
from parameters import Parameters
parameters = Parameters()
board_size = parameters.board_size


class ShortestPathBot:
    def __init__(self, playing, graph):

        """
        A bot which plays quoridor by always moving in the direction which will fastest get to the destination

        inputs:
            playing: int
                integer representing the player playing the game

            graph: Graph
                graph of paths
        """
        self.playing = playing
        self.graph = graph

    def move(self, board, graph):
        player_1_loc = np.where(board[:, :, 2] == 1)
        player_2_loc = np.where(board[:, :, 3] == 1)
        move = graph.get_sp_move(player_1_loc, player_2_loc, self.playing - 1)
        return move
