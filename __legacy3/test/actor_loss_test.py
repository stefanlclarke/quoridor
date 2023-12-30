import numpy as np
from loss_functions.actor_loss import actor_loss_fn
import torch


def test_actor_loss():
    action_probs = torch.tensor([0.3, 0.9, 0.5])
    distributions = torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.5, 0.5]])
    advantage = torch.tensor([1., -1., 3.])

    entropy_constant = 1
    entropy_bias = 0

    loss, entropy_loss = actor_loss_fn(action_probs, distributions, advantage, entropy_constant, entropy_bias)

    target_loss = -torch.sum(torch.tensor([np.log(0.3) * 1., np.log(0.9) * (-1), np.log(0.5) * 3]))

    assert torch.abs(target_loss - loss) < 1e-4