import torch
from parameters import Parameters

parameters = Parameters()
epsilon = parameters.ppo_epsilon

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def actor_loss(memory, advantages, entropy_constant=0.001):
    loss = torch.tensor(0.).to(device)
    for i in range(len(advantages)):
        advantage = advantages[i].detach()
        probs = [x.unsqueeze(0) for x in memory.game_log[i][4][2]]
        print(probs)
        probabilities = torch.cat(probs)
        prob_denominator = probabilities.detach()
        distributions = memory.game_log[i][4][3]
        r = probabilities / prob_denominator
        print('r')
        print(r)
        term_1 = torch.sum(r * advantage)
        term_2 = torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantage
        loss += torch.sum(torch.minimum(term_1, term_2))

        entropy = 0.
        for j in range(len(distributions)):
            distribution = distributions[j]
            entropy += - torch.sum(distribution * torch.log(distribution))

        loss += - entropy_constant * entropy
    return loss
