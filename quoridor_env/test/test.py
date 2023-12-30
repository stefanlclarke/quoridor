from quoridor_env.gym_env import LoadedQuoridorGym
from quoridor_env.agents.shortest_path_agent import ShortestPathAgent


def test_env():
    opponent = ShortestPathAgent(0)
    gym_env = LoadedQuoridorGym(opponent=opponent)

    gym_env.render()