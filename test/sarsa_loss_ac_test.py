from loss_functions.sarsa_loss_ac import get_sarsa_ac
import torch
import numpy as np


def test_sarsa_ac():
    state_values = torch.tensor([1., 2., 3.])
    rewards = torch.tensor([0., -1., 3.])
    lambd = 0.7
    gamma = 0.9

    sarsa_values = get_sarsa_ac(state_values, rewards, lambd, gamma)

    correct_sarsa_values = np.zeros(3)
    correct_sarsa_values[0] = (rewards[0] + gamma * state_values[1]) \
        + lambd * (rewards[0] + gamma * rewards[1] + gamma**2 * state_values[2]) \
        + lambd**2 * (rewards[0] + gamma * rewards[1] + gamma**2 * rewards[2])
    correct_sarsa_values[0] = correct_sarsa_values[0] / (1 + lambd + lambd**2)

    correct_sarsa_values[1] = (rewards[1] + gamma * state_values[2]) \
        + lambd * (rewards[1] + gamma * rewards[2])
    correct_sarsa_values[1] = correct_sarsa_values[1] / (1 + lambd)

    correct_sarsa_values[2] = (rewards[2])

    for i in range(len(sarsa_values)):
        assert np.abs(correct_sarsa_values[i] - sarsa_values[i]) < 1e-4