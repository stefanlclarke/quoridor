import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def actor_loss(memory, advantages, entropy_constant=0.001):
    loss = torch.tensor(0.)
    for i in range(len(advantages)):
        advantage = advantages[i].detach().to(device)
        probs = [x.unsqueeze(0) for x in memory.game_log[i][4][2]]
        probabilities = torch.cat(probs)
        distributions = memory.game_log[i][4][3]
        log_probabilities = torch.log(probabilities)
        loss += - torch.sum(log_probabilities * advantage)

        entropy = 0.
        for j in range(len(distributions)):
            distribution = distributions[j]
            entropy += - torch.sum(distribution * torch.log(distribution))

        loss += - entropy_constant * entropy
    return loss
