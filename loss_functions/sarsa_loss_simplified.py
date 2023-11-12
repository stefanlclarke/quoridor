import torch
import numpy as np
from parameters import Parameters

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

parameters = Parameters()
gamma = parameters.gamma
lambd = parameters.lambd


def sarsa_loss(memory, net, epoch, possible_moves, printing=False, return_advantage=False):

    """
    calculates the loss on memory given network net

    inputs:
        memory: Memory
            location of saved game memory

        net: NN
            the neural net to be trained

        epoch: int
            the epoch of training we are in

        possible_moves: list
            the list of all moves which can be made

        printing: bool
            True if you want to print details

        return_advantage: bool
            True if you want to return the advantage as well as the loss

    returns:
        loss: torch.tensor (and optionally advantage: torch.tensor)

    """
    loss = torch.tensor([0.]).to(device)
    advantages = []
    for game in memory.game_log:
        states = game[0]
        actions = game[1]
        rewards = game[2]
        if epoch == 0:
            move_avs = game[4][0]
            avs = game[4][1]
            move_avs = game[4][1]
            random_moves = game[3]
        else:
            move_avs = []
            avs = []
            random_moves = []
            for i in range(len(states)):
                state_actions = [np.concatenate([states[i], move]) for move in possible_moves]
                values = torch.cat([net.forward(torch.from_numpy(sa).float().to(device)) for sa in state_actions])
                best_move_value = torch.max(values)
                action_index = np.where(actions[i].flatten() == 1)[0]
                move_value = values[action_index]
                move_avs.append(move_value.unsqueeze(0))
                avs.append(best_move_value.unsqueeze(0))
                random_moves.append(True)

        sarsa_values = np.zeros(len(rewards))
        discounted_rewards = [gamma**u * rewards[u] for u in range(len(rewards))]

        # iterate over steps of the game
        for i in range(len(sarsa_values)):

            # iterate over steps j after step i
            for j in range(len(sarsa_values) - i - 1):

                # get this-step sarsa
                discounted_step_reward = sum(discounted_rewards[i:i + j + 1]) / (gamma**i) + avs[i + j + 1] \
                    * gamma ** (j + 1)
                sarsa_ij = lambd**j * (discounted_step_reward)
                sarsa_values[i] += sarsa_ij

            # get terminal sarsa
            discounted_step_reward = sum(discounted_rewards[i:]) / (gamma**i)
            sarsa_ij = lambd**(len(sarsa_values) - i - 1) * (discounted_step_reward)
            sarsa_values[i] += sarsa_ij

            # normalize
            sarsa_values[i] *= (1 - lambd) / (1 - lambd**(len(sarsa_values) - i))

        # calculate advantage and loss
        if len(move_avs) > 0:
            advantage = torch.from_numpy(sarsa_values).to(device) - torch.cat(avs).reshape(sarsa_values.shape)
            loss = loss + torch.pow(advantage, 2).mean()
            advantages.append(advantage)
        if printing:
            print('iteration')
            print('rewards', rewards)
            print('sarsa', sarsa_values)
            print('random', random_moves)
            print('avs', avs)
            print('move avs', move_avs)
            print('advantages', advantages)

    if return_advantage:
        return loss, advantages

    return loss
