from dataclasses import dataclass


@dataclass
class Configuration:

    # dictates whether we are testing or not
    PLAY_QUORIDOR = True
    ZERO_CRITIC = False

    # parameters for the game
    BOARD_SIZE = 3
    NUMBER_OF_WALLS = 1

    # dimensions of the neural networks
    ACTOR_NUM_HIDDEN = 3
    ACTOR_SIZE_HIDDEN = 64
    CRITIC_NUM_HIDDEN = 3
    CRITIC_SIZE_HIDDEN = 64
    SOFTMAX_REGULARIZER = 0.9

    # conv stuff
    CONV_SIDELEN = 3
    CONV_INTERNAL_CHANNELS = 8
    CONV_NUM_LAYERS = 2
    USE_CONV_NET = False
    CONV_KERNEL_SIZE = 1

    # parameters relating to the actor playing the game
    ILLEGAL_MOVE_REWARD = -0.25
    LEGAL_MOVE_REWARD = 0
    WIN_REWARD = 1000.
    MAX_ROUNDS_PER_GAME = 40
    EPSILON = 0.4
    EPSILON_DECAY = 0.99
    MOVE_PROB = 0.4
    FORWARD_PROB = 0.4
    MOVE_PROB_DECAY = 1.
    GAMMA = 0.95
    LAMBD = 0.5
    MINIMUM_EPSILON = 0.1
    MINIMUM_MOVE_PROB = 0.4
    CUT_AT_RANDOM_MOVE = False
    RANDOM_PROPORTION = 0.4
    WIN_SPEED_PARAMETER = 2

    # learning rates
    Q_LEARNING_RATE = 0.004
    ACTOR_LEARNING_RATE = 0.004
    CRITIC_LEARNING_RATE = 0.004

    # clipping
    MAX_GRAD_NORM = 10000.
    ACTOR_WEIGHT_CLIP = 1000
    ENTROPY_CONSTANT = 0.003
    ENTROPY_BIAS = 1

    # what am I training?
    TRAIN_ACTOR = True
    TRAIN_CRITIC = True
    N_ITERATIONS_ONLY_CRITIC = 0
    ITERATIONS_ONLY_ACTOR_TRAIN = 0

    # epoch parameters
    WORKER_GAMES_BETWEEN_TRAINS = 16
    TOTAL_EPOCHS = 10000000000
    SAVE_EVERY = 40
    N_CORES = 1
    EPOCHS_PER_WORKER = 100
    TOTAL_RESET_EVERY = None
    PRINT_EVERY = 3
    DECREASE_EPSILON_EVERY = 10

    # self-play parameters
    OLD_SELFPLAY = True
    LOAD_FROM_LAST = 100
    RELOAD_EVERY = 20
    RELOAD_DISTRIBUTION = "geometric"

    # continue a previous run
    CONTINUE = None


def update_config_from_yaml(config, cfg_yaml):

    for key in dir(config):
        if key[0] != '_':
            print('setting attr {}'.format(key))
            setattr(config, key, cfg_yaml[key])


config = Configuration()