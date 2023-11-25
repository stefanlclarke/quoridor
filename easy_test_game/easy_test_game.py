import numpy as np
from parameters import Parameters

parameters = Parameters()
BOARD_SIZE = parameters.board_size
START_WALLS = parameters.number_of_walls


class TestGame:
    
    def __init__(self):

        self.board_size = BOARD_SIZE
        self.board = np.zeros((BOARD_SIZE, 4, 4))
        self.n_rounds = 0
        self.playing = 0

    def scramble_board(self):
        c1 = np.random.choice(self.board.shape[0])
        c2 = np.random.choice(self.board.shape[1])
        c3 = np.random.choice(self.board.shape[2])

        self.board = self.board * 0
        self.board[c1, c2, c3] = 1

    def move(self, move, get_time_info=True):
        reward = np.linalg.norm(move - self.board)
        self.scramble_board()
        new_state = self.get_state(flatten=True)
        self.playing = self.playing + 1 % 2
        winner = 0

        return new_state, self.playing, winner, reward, True, move, 0, 0, 0, 0

    def get_state(self, flip=False, flatten=True):

        if flatten:
            return self.board.flatten()

        return self.board