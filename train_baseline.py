from stable_baselines3 import PPO
from quoridor_env.gym_env import LoadedQuoridorGym
from quoridor_env.agents.shortest_path_agent import ShortestPathAgent

env = LoadedQuoridorGym(opponent=ShortestPathAgent(0))

print(env.action_space)
print(env.observation_space)
print(env.get_state())

model = PPO("MlpPolicy", env, verbose=2, policy_kwargs={'net_arch': [256, 256, 256]}, gamma=0.4, learning_rate=0.003,
            ent_coef=1.)

model.learn(total_timesteps=400000, log_interval=2, progress_bar=True)

# save the model
model.save("baselines/models/PPO_example5")