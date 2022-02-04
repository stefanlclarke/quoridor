import numpy as np
from game.game_helper_functions import *
from parameters import Parameters
from game.printing import get_printable_board
import copy
parameters = Parameters()
board_size = parameters.board_size

def create_map(board, location):
    visited = np.zeros((board_size, board_size)) + board_size**2
    visited[int(location[0]), int(location[1])] = 1
    moves = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.])]
    checking = True
    i = 1
    most_recent_additions = [location]
    new_additions = []
    while checking:
        i += 1
        for point in most_recent_additions:
            possible_moves = []
            for move in moves:
                if (point+move>=0).all() and (point+move<board_size).all():
                    possible_moves.append(move)
            for move in possible_moves:
                if visited[int(point[0] + move[0]), int(point[1] + move[1])] == board_size**2:
                    move_legal, jump = check_move_legal(board, point, move, False, True)
                    if move_legal:
                        new_additions.append(point + move)
                        visited[int(point[0]+move[0]), int(point[1]+move[1])] = i

        most_recent_additions = new_additions
        new_additions = []
        if len(most_recent_additions) == 0:
            checking = False
    return visited

def shortest_path_to_end(board, location, player=2):
    map = create_map(board, location)
    if player == 2:
        start_loc = np.array([0, np.argmin(map[0])])
    else:
        start_loc = np.array([board_size-1, np.argmin(map[-1])])
    i = map[int(start_loc[0]), int(start_loc[1])]
    point = start_loc
    moves = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.])]
    checking = True
    route = []
    while checking:
        new_check = True
        if i == 1:
            checking = False
            break
        possible_moves = []
        for move in moves:
            if (point+move>=0).all() and (point+move<board_size).all() and new_check:
                legal, jump = check_move_legal(board, point, move, False, True)
                if map[int(point[0] + move[0]), int(point[1] + move[1])] == i - 1 and legal and new_check:
                    route.append(copy.copy(point))
                    i = i-1
                    point = point + move
                    new_check = False
    return route

class ShortestPathBot:
    def __init__(self, playing):
        self.playing = playing

    def move(self, board):

        current_loc = np.where(board[:,:,self.playing+1]==1)
        pos = np.array([current_loc[0][0], current_loc[1][0]])
        move =  shortest_path_to_end(board, pos, self.playing)[-1] - pos
        return np.array([move[0], move[1], 0, 0, 0, 0])

def get_rollout_winner(game):
    bots = [ShortestPathBot(1), ShortestPathBot(2)]
    playing = True
    while playing:
        moving = game.moving_now
        move = bots[moving].move(game.get_state(flatten=False)[0])
        new_state, playing, winner, reward, legal = game.move(move)
        if winner != 0:
            return winner
