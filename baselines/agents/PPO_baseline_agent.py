import numpy as np
from quoridor_env.agents.template_agent import QuoridoorAgent
from stable_baselines3 import PPO
from quoridor_env.config import game_config


class PPOBaselineAgent(QuoridoorAgent):

    def __init__(self, model_save_loc, env):
        QuoridoorAgent.__init__(self)

        self.model = PPO.load(model_save_loc, env=env)
        self.playing = 0

    def move(self, input):
        move_int = self.model.predict(input, deterministic=True)
        move = np.zeros(game_config.INPUT_DIM)
        move[move_int] = 1
        return move