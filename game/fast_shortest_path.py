import numpy as np
from game.game_helper_functions import *
from parameters import Parameters
from game.printing import get_printable_board
import copy
parameters = Parameters()
board_size = parameters.board_size

class ShortestPathBot:
    def __init__(self, playing, graph):
        self.playing = playing
        self.graph = graph

    def move(self, board, graph):

        current_loc = np.where(board[:,:,self.playing+1]==1)
        pos = np.array([current_loc[0][0], current_loc[1][0]])
        next_move = list(graph.direction_graph[self.playing-1][tuple(pos)].keys())[0]
        next_move = np.array(next_move) - pos
        #graph.plot()
        #graph.player_plot(0)
        #graph.player_plot(1)
        move = np.array(next_move)
        return np.array([move[0], move[1], 0, 0, 0, 0])
