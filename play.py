from quoridor_env.pygame_.pygame_player import PygamePlayer
from quoridor_env.agents.random_agent import RandomAgent
from quoridor_env.agents.shortest_path_agent import ShortestPathAgent

spagent = ShortestPathAgent(0)
randomagent = RandomAgent()

player = PygamePlayer(agent_1=spagent, agent_2="human")

spagent.game_graph = player.game.board_graph
player.play()
