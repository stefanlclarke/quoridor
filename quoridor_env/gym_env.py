import gym
from game.game import Quoridor
from game.game_config import game_config
import numpy as np


class LoadedQuoridorGym(gym.Env, Quoridor):

    def __init__(self, opponent):

        Quoridor.__init__(board_size=game_config.BOARD_SIZE,
                          start_walls=game_config.NUMBER_OF_WALLS,
                          p1_start=None, p2_start=None,
                          legal_move_reward=0.,
                          illegal_move_reward=-0.25)

        input_dim = game_config.BOARD_SIZE**2 * 4 + 2 * (game_config.NUMBER_OF_WALLS + 1)
        output_dim = 4 + 2 * (game_config.BOARD_SIZE - 1)**2

        self.observation_space = gym.spaces.MultiDiscrete([1 for _ in range(input_dim)])
        self.action_space = gym.spaces.MultiDiscrete([1 for _ in range(output_dim)])

        self.opponent = opponent

        self.player_team = np.random.choice(2)

        if self.player_team == 0:
            self.flip = False
        else:
            self.flip = True

        if self.player_team == 1:
            self.opponent_move()

    def opponent_move(self):
        new_state_opponent = self.get_state(flip=not self.flip, flatten=True)
        opponent_move = self.opponent.move(new_state_opponent)
        self.move(opponent_move, reformat_from_onehot=True, flip=not self.flip)

    def step(self, action):

        if self.player_team == 0:
            flip = False
        else:
            flip = True

        result = self.move(action, reformat_from_onehot=True, flip_reformat=flip)

        if not self.playing:
            return self.get_state(flip=flip, flatten=True), game_config.WIN_REWARD, True, {}

        self.opponent_move()

        if not self.playing:
            return self.get_state(flip=flip, flatten=True), - game_config.WIN_REWARD, True, {}

        reward = result[3]
        return self.get_state(flip=flip, flatten=True), reward, False, {}

    def render(self, mode="human"):
        self.print()

    def reset(self):
        self.reset_board()
        self.player_team = np.random.choice(2)

        if self.player_team == 0:
            self.flip = False
        else:
            self.flip = True

        if self.player_team == 1:
            self.opponent_move()
