from dataclasses import dataclass


@dataclass
class Configuration:

    # parameters for the game itself
    BOARD_SIZE = 3
    NUMBER_OF_WALLS = 1

    # environment parameters
    RANDOM_PROPORTION = 0.5
    MAX_STEPS = 100

    # reward parameters for training
    WIN_REWARD = 1000
    ILLEGAL_MOVE_REWARD = -0.25

    # calculations
    INPUT_DIM = BOARD_SIZE**2 * 4 + 2 * (NUMBER_OF_WALLS + 1)
    OUTPUT_DIM = 4 + 2 * (BOARD_SIZE - 1)**2


game_config = Configuration()
