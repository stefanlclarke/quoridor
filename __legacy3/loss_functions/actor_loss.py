import torch

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
            probs = [x.unsqueeze(0) for x in memory.game_log[i][4][1]]
            probabilities = torch.cat(probs)
            distributions = memory.game_log[i][4][2]
            distributions = torch.stack(distributions)
        else:
            raise ValueError("Only supported for epoch 0!")

        loss_i, entropy_i = actor_loss_fn(probabilities, distributions, advantage, entropy_constant, entropy_bias)
        loss += loss_i
        entropy_loss += entropy_i

    return loss, entropy_loss


def actor_loss_fn(action_probs, action_distributions, advantage, entropy_constant, entropy_bias):

    assert action_probs.shape[0] == action_distributions.shape[0]
    assert advantage.shape[0] == action_probs.shape[0]

    log_probabilities = torch.log(action_probs)
    loss = - torch.sum(log_probabilities * advantage)

    entropy = 0.
    for j in range(len(action_distributions)):
        distribution = action_distributions[j]
        entropy += - torch.sum(distribution * torch.log(distribution))

    entropy_loss = torch.minimum(- entropy_constant * (entropy + entropy_bias), torch.tensor(0.))

    return loss, entropy_loss