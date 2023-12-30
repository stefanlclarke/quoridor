from dataclasses import dataclass


@dataclass
class Configuration:
    BOARD_SIZE = 3
    NUMBER_OF_WALLS = 1
    RANDOM_PROPORTION = 0.5


game_config = Configuration()
