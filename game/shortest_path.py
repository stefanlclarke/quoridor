import numpy as np


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

        current_loc = np.where(board[:, :, self.playing + 1] == 1)
        pos = np.array([current_loc[0][0], current_loc[1][0]])
        next_move = list(graph.direction_graph[self.playing - 1][tuple(pos)].keys())[0]
        next_move = np.array(next_move) - pos
        move = np.array(next_move)
        return np.array([move[0], move[1], 0, 0, 0, 0])
