from quoridor_env.gym_env import LoadedQuoridorGym
from quoridor_env.agents.random_agent import RandomAgent


def get_env():
    env = LoadedQuoridorGym(opponent=RandomAgent())
    return env
