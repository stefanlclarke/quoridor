import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def sarsa_loss_ac(memory, net, epoch, possible_moves, lambd, gamma, printing=False, return_advantage=False):

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
        rewards = game[2]
        if epoch == 0:
            state_values = game[4][0]
            random_moves = game[3]
        else:
            state_values = []
            random_moves = []
            for i in range(len(states)):
                state = states[i]
                value = net.forward(torch.from_numpy(state).float().to(device))
                state_values.append(value)
                random_moves.append(True)

        sarsa_values = get_sarsa_ac(state_values, rewards, lambd, gamma)

        # calculate advantage and loss
        if len(state_values) > 0:
            advantage = torch.from_numpy(sarsa_values).to(device) - torch.cat(state_values).reshape(sarsa_values.shape)
            loss = loss + torch.pow(advantage, 2).mean()
            advantages.append(advantage)
        if printing:
            print('iteration')
            print('rewards', rewards)
            print('sarsa', sarsa_values)
            print('random', random_moves)
            print('state values', state_values)
            print('advantages', advantages)

    if return_advantage:
        return loss, advantages

    return loss


def get_sarsa_ac(state_values, rewards, lambd, gamma):
    sarsa_values = np.zeros(len(rewards))
    discounted_rewards = [gamma**u * rewards[u] for u in range(len(rewards))]

    # iterate over steps of the game
    for i in range(len(sarsa_values)):

        # iterate over steps j after step i
        for j in range(len(sarsa_values) - i - 1):

            # get this-step sarsa
            discounted_step_reward = sum(discounted_rewards[i:i + j + 1]) / (gamma**i) + state_values[i + j + 1] \
                * gamma ** (j + 1)
            sarsa_ij = lambd**j * (discounted_step_reward)
            sarsa_values[i] += sarsa_ij

        # get terminal sarsa
        discounted_step_reward = sum(discounted_rewards[i:]) / (gamma**i)
        sarsa_ij = lambd**(len(sarsa_values) - i - 1) * (discounted_step_reward)
        sarsa_values[i] += sarsa_ij

        # normalize
        sarsa_values[i] *= (1 - lambd) / (1 - lambd**(len(sarsa_values) - i))

    return sarsa_values