import numpy as np


class Parameters:
    def __init__(self):

        """
        A storage class for game parameters
        """

        # dictates whether we are testing or not
        self.play_quoridor = False
        self.zero_critic = False

        # parameters for the game
        self.board_size = 2
        self.number_of_walls = 1

        # dimensions of the neural networks
        self.bot_in_dimension = ((self.board_size**2) * 4 + 2 * (self.number_of_walls + 1))
        self.bot_out_dimension = 4 + 2 * (self.board_size - 1)**2
        self.actor_num_hidden = 3
        self.actor_size_hidden = 64
        self.critic_num_hidden = 3
        self.critic_size_hidden = 64
        self.softmax_regularizer = 0.9

        # conv stuff
        self.sidelen = self.board_size
        self.conv_internal_channels = 8
        self.num_conv = 2
        self.convolutional = False
        self.conv_kernel_size = 1

        # parameters relating to the actor playing the game
        self.illegal_move_reward = -10 / 40
        self.legal_move_reward = 0
        self.win_reward = 0 #100 / 100
        self.max_rounds_per_game = 12 #40
        self.epsilon = 0.4
        self.epsilon_decay = 0.985
        self.move_prob = 0.4
        self.forward_prob = 0.4
        self.move_prob_decay = 1.
        self.gamma = 0.95
        self.lambd = 0. #0.5
        self.minimum_epsilon = 0.07
        self.minimum_move_prob = 0.4
        self.cut_at_random_move = False
        self.random_proportion = 0.4
        self.win_speed_param = 2

        # learning rates
        self.learning_rate = 0.004
        self.actor_learning_rate = 0.004
        self.critic_learning_rate = 0.04
        self.actor_weight_clip = 1e3

        # clipping
        self.max_grad_norm = 10000.
        self.entropy_constant = 0.003

        # what am I training?
        self.train_actor = True
        self.train_critic = True
        self.n_iterations_only_critic = 1000

        # epoch parameters
        self.games_between_backprops = 8
        self.backprops_per_worker = 4
        self.epochs = 10000000
        self.save_every = 20
        self.n_cores = 4
        self.backwards_per_worker = 1
        self.total_reset_every = 4000

        # storage parameters
        self.save_game_every = 100

        # stores a set of possible moves
        self.possible_moves = [np.zeros(self.bot_out_dimension) for _ in range(self.bot_out_dimension)]
        for i in range(self.bot_out_dimension):
            self.possible_moves[i][i] = 1
