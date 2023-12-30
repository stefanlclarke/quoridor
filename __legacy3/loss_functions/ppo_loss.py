import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def ppo_actor_loss(memory, advantages, epsilon=0.1, epoch=0, net=None):
    loss = torch.tensor(0.).to(device)
    for i in range(len(advantages)):
        advantage = advantages[i].detach()

        if epoch == 0:
            probs = [x.unsqueeze(0) for x in memory.game_log[i][4][1]]
        else:
            actions = [x for x in memory.game_log[i][1]]
            states = [x for x in memory.game_log[i][0]]
            action_ind = [np.where(x == 1)[0] for x in actions]

            probs = []
            for j in range(len(states)):
                state_torch = torch.from_numpy(states[j]).to(device).float()
                move, pties, pty = net.move(state_torch)
                action_pty = pties[action_ind[j]]
                probs.append(action_pty)

        probabilities = torch.cat(probs)
        prob_denominator = probabilities.detach()
        r = probabilities / prob_denominator
        term_1 = r * advantage
        term_2 = torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantage
        loss += torch.sum(torch.minimum(term_1, term_2))
    return - loss
