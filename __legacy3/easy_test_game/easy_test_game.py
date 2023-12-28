import numpy as np


class EasyGame:

    def __init__(self, board_size, number_of_walls, max_moves):

        self.board_size = board_size
        self.start_walls = number_of_walls
        self.max_moves = max_moves
        self.board = np.zeros((board_size, board_size, 4))
        self.n_rounds = 0
        self.moving_now = 0
        self.playing = True
        self.max_moves = self.max_moves
        self.move_size = 4 + 2 * (self.board_size - 1)**2
        self.scramble_board()

    def scramble_board(self):
        c1 = np.random.choice(self.move_size)
        board = self.board.flatten()
        board = board * 0
        board[c1] = 1
        board = board.reshape((self.board_size, self.board_size, 4))
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
            return np.block([[self.board.flatten(), np.zeros(2 + 2 * self.start_walls)]]).flatten()

        return self.board

    def reset(self, random_positions=True):
        self.__init__()

    def print(self):
        print(np.where(self.board.flatten() == 1))