import numpy as np
from parameters import Parameters

parameters = Parameters()
BOARD_SIZE = parameters.board_size
START_WALLS = parameters.number_of_walls
MAX_MOVES = parameters.max_rounds_per_game - 4
MOVE_SIZE = 4 + 2 * (BOARD_SIZE - 1)**2


class EasyGame:

    def __init__(self):

        self.board_size = BOARD_SIZE
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, 4))
        self.n_rounds = 0
        self.moving_now = 0
        self.playing = True
        self.max_moves = MAX_MOVES
        self.scramble_board()

    def scramble_board(self):
        c1 = np.random.choice(MOVE_SIZE)
        board = self.board.flatten()
        board = board * 0
        board[c1] = 1
        board = board.reshape((BOARD_SIZE, BOARD_SIZE, 4))
        self.board = board

    def move(self, move, get_time_info=True, reformat_from_onehot=False, flip_reformat=False):
        reward = - np.linalg.norm(move - self.get_state(flatten=True)[0:move.size])
        self.scramble_board()
        new_state = self.get_state(flatten=True)
        self.moving_now = (self.moving_now + 1) % 2
        winner = 0

        if self.n_rounds > self.max_moves:
            winner = np.random.choice(2) * 2 - 1
            self.playing = False

        self.n_rounds += 1

        return new_state, self.playing, winner, reward, True, move, 0, 0, 0, 0

    def get_state(self, flip=False, flatten=True):

        if flatten:
            return np.block([[self.board.flatten(), np.zeros(2 + 2 * START_WALLS)]]).flatten()

        return self.board

    def reset(self, random_positions=True):
        self.__init__()

    def print(self):
        print(np.where(self.board.flatten() == 1))