class Parameters:
    def __init__(self):
        self.board_size = 5
        self.number_of_walls = 5
        self.bot_in_dimension = ((self.board_size**2)*4 + 2*(self.number_of_walls+1))
        self.bot_out_dimension = 4 + 2 * (self.board_size - 1)**2

        self.illegal_move_reward = -10/400
        self.legal_move_reward = 0
        self.win_reward = 100/100

        self.max_rounds_per_game = 40

        self.actor_num_hidden = 2
        self.actor_size_hidden = 128
        self.critic_num_hidden = 2
        self.critic_size_hidden = 128

        self.epsilon = 0.07
        self.epsilon_decay = 0.985
        self.move_prob = 0.4
        self.forward_prob = 0.4
        self.move_prob_decay = 1.
        self.gamma = 0.95
        self.lambd = 0.5

        self.learning_rate = 0.004
        self.actor_learning_rate = 0.0004
        self.critic_learning_rate = 0.001

        self.minimum_epsilon = 0.07
        self.minimum_move_prob = 0.4
        self.cut_at_random_move = True

        self.random_proportion = 0.8

        self.games_per_iter = 5

        self.max_grad_norm = 1.
        self.entropy_constant = 1.
