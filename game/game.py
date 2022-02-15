import numpy as np
import copy
import time
from game.game_helper_functions_parallel import *
from parameters import Parameters
from game.printing import get_printable_board

parameters = Parameters()
board_size = parameters.board_size
start_walls = parameters.number_of_walls


def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

class Player:
    def __init__(self, start_pos, number_of_walls):
        self.start_pos = start_pos
        self.start_walls = number_of_walls
        self.walls = number_of_walls
        self.pos = start_pos

class Quoridor:
    def __init__(self, p1_start=np.array([0, board_size//2]), p2_start=np.array([board_size-1, board_size//2]), get_time_info=False):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size, 4))
        self.players = [Player(p1_start, start_walls), Player(p2_start, start_walls)]
        self.num_players = len(self.players)
        self.board[p1_start[0], p1_start[1], 2] = 1
        self.board[p2_start[0], p2_start[1], 3] = 1
        self.moving_now = 0
        self.playing = True
        self.start_walls = parameters.number_of_walls
        self.winner = 0

    def move(self, command, get_time_info=False):
        moving = 0.
        illegal_move_handling = 0.
        checking_winner = 0.
        wall_handling = 0.

        t0 = time.time()
        if not self.playing:
            return None

        player_moving = self.players[self.moving_now]
        player_moving_pos = player_moving.pos
        other_player_pos = self.players[((self.moving_now + 1) % 2)].pos
        move_command = command[0:2]
        wall_pos_command = command[2:4]
        wall_orientation_command = command[4:6]
        legal_move, jump = move_piece(self.board, player_moving_pos, move_command)
        reward = 0

        t1 = time.time()
        if legal_move:
            if jump:
                self.players[self.moving_now].pos = 2*move_command + player_moving_pos
            else:
                self.players[self.moving_now].pos = move_command + player_moving_pos

        if player_moving.walls > 0:
                legal_wall = place_wall_with_check(self.board, wall_pos_command, wall_orientation_command, self.players[0].pos, self.players[1].pos)
                if legal_wall:
                    player_moving.walls -= 1
        else:
            legal_wall = False

        if legal_move or legal_wall:
            legal = True
            reward += parameters.legal_move_reward
        else:
            reward += parameters.illegal_move_reward
            legal = False

        t2 = time.time()
        if not legal:
            possible_moves = get_legal_moves(self.board, player_moving_pos)

            if len(possible_moves) == 0:
                self.playing = False
                self.winner = 0
                return self.get_state(), self.playing, self.winner, reward, False

            random_move_ind = np.random.choice(len(possible_moves))
            random_move = possible_moves[random_move_ind]
            true, jump = move_piece(self.board, player_moving_pos, random_move)

            if true:
                if jump:
                    self.players[self.moving_now].pos = 2*random_move + player_moving_pos
                else:
                    self.players[self.moving_now].pos = random_move + player_moving_pos

        t3 = time.time()
        winner = check_win(self.board)
        if winner != 0:
            self.playing = False
            self.winner = winner

        self.moving_now = (self.moving_now + 1) % 2

        t4 = time.time()

        moving += t1 - t0
        wall_handling += t2 - t1
        illegal_move_handling += t3 - t2
        checking_winner += t4 - t3

        if not get_time_info:
            return self.get_state(), self.playing, winner, reward, legal
        else:
            return self.get_state(), self.playing, winner, reward, legal, moving, illegal_move_handling, checking_winner, wall_handling

    def get_state(self, flip=False, flatten=True):
        p1_walls = np.zeros(self.start_walls + 1)
        p2_walls = np.zeros(self.start_walls + 1)
        p1_walls[self.players[0].walls] = 1
        p2_walls[self.players[1].walls] = 1
        if not flip:
            if flatten:
                flat_board = copy.copy(self.board).flatten()
                return np.concatenate([flat_board, p1_walls, p2_walls])
            else:
                return self.board, p1_walls, p2_walls
        else:
            if flatten:
                flat_board = flip_board(self.board, self.players[0].pos, self.players[1].pos).flatten()
                return np.concatenate([flat_board, p2_walls, p1_walls])
            else:
                return flip_board(self.board, self.players[0].pos, self.players[1].pos), p1_walls, p2_walls

    def reset(self, random_positions=False):
        if random_positions:
            p1_start = np.array([int(np.random.choice(int(np.floor(board_size/2)))), int(np.random.choice(int(board_size)))])
            p2_start = np.array([int(board_size - 1 - np.random.choice(int(np.floor(board_size/2)))), int(np.random.choice(int(board_size)))])
            self.__init__(p1_start, p2_start)
            self.randomly_place_walls()
        else:
            self.__init__()

    def copy_game(self, other_game):
        self.board_size = other_game.board_size
        self.board = copy.copy(other_game.board)
        self.players = [copy.deepcopy(player) for player in other_game.players]
        self.num_players = len(self.players)
        self.moving_now = other_game.moving_now
        self.playing = other_game.playing

    def copy_board(self, board, player_1_loc, player_2_loc, player_1_walls, player_2_walls):
        self.board = copy.copy(board)
        self.players = [Player(player_1_loc, player_1_walls), Player(player_2_loc, player_2_walls)]
        self.num_players = len(self.players)
        self.moving_now = 0
        self.playing = True

    def print(self):
        get_printable_board(self.board, self.players[0].walls, self.players[1].walls)

    def randomly_place_walls(self):
        number_of_walls_from_1 = np.random.choice(self.start_walls)
        number_of_walls_from_2 = np.random.choice(self.start_walls)

        for i in range(number_of_walls_from_1):
            loc = np.array([np.random.choice(self.board_size), np.random.choice(self.board_size)])
            u = np.random.uniform()
            if u > 0.5:
                orientation = np.array([0.,1.])
            else:
                orientation = np.array([1.,0.])
            legal = place_wall_with_check(self.board, loc, orientation, self.players[0].pos, self.players[1].pos)
            if legal:
                self.players[0].walls -= 1

        for i in range(number_of_walls_from_2):
            loc = np.array([np.random.choice(self.board_size), np.random.choice(self.board_size)])
            u = np.random.uniform()
            if u > 0.5:
                orientation = np.array([0.,1.])
            else:
                orientation = np.array([1.,0.])
            legal = place_wall_with_check(self.board, loc, orientation, self.players[0].pos, self.players[1].pos)
            if legal:
                self.players[1].walls -= 1
