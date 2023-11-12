import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def actor_loss(memory, advantages,
               entropy_constant=0.001, entropy_bias=100, epoch=0):
    loss = torch.tensor(0.).to(device)
    entropy_loss = torch.tensor(0.).to(device)
    for i in range(len(advantages)):
        advantage = advantages[i].detach()
        if epoch == 0:
            probs = [x.unsqueeze(0) for x in memory.game_log[i][4][2]]
            probabilities = torch.cat(probs)
            distributions = memory.game_log[i][4][3]
        else:
            raise ValueError('Epoch cannot be nonzero!')

        log_probabilities = torch.log(probabilities)
        loss += - torch.sum(log_probabilities * advantage)

        entropy = 0.
        for j in range(len(distributions)):
            distribution = distributions[j]
            entropy += - torch.sum(distribution * torch.log(distribution))

        entropy_loss += torch.minimum(- entropy_constant * (entropy + entropy_bias), torch.tensor(0.))

    return loss, entropy_loss
