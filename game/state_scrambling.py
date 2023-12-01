import numpy as np
import copy


def mild_continuous_scramble(board, sigma=0.1):
    scrambled_board = copy.copy(board)
    noise = np.random.normal(size=scrambled_board.size, scale=sigma).reshape(scrambled_board.shape)
    scrambled_board += noise
    return scrambled_board
