from quoridor_env.gym_env import LoadedQuoridorGym
from quoridor_env.agents.shortest_path_agent import ShortestPathAgent
from quoridor_env.agents.random_agent import RandomAgent
from quoridor_env.config import game_config


def test_env():
    opponent = ShortestPathAgent(0)
    you = RandomAgent()
    gym_env = LoadedQuoridorGym(opponent=opponent)

    done = False

    while not done:
        state, reward, done, info = gym_env.step(you.move(gym_env.get_state()))

    assert abs(reward) == game_config.WIN_REWARD

    assert state.size == game_config.BOARD_SIZE**2 * 4 + 2 * (game_config.NUMBER_OF_WALLS + 1)

    assert you.move(gym_env.get_state()).size == 4 + 2 * (game_config.BOARD_SIZE - 1)**2
