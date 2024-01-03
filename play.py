from quoridor_env.pygame_.pygame_player import PygamePlayer
from quoridor_env.agents.random_agent import RandomAgent
from quoridor_env.agents.shortest_path_agent import ShortestPathAgent
from baselines.agents.PPO_baseline_agent import PPOBaselineAgent
from quoridor_env.gym_env import LoadedQuoridorGym


spagent = ShortestPathAgent(0)
randomagent = RandomAgent()
ppoagent = PPOBaselineAgent(model_save_loc='/Users/stefanclarkework/Desktop/quoridor/baselines/models/PPO_example5.zip',
                            env=LoadedQuoridorGym(opponent=randomagent))

player = PygamePlayer(agent_1="human", agent_2=ppoagent)

player.play()
