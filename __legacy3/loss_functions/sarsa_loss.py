import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def sarsa_loss(memory, net, epoch, possible_moves, gamma, lambd, printing=False, return_advantage=False,
               cut_at_random_move=True):

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

    cutting = cut_at_random_move

    loss = torch.tensor([0.]).to(device)
    advantages = []
    for game in memory.game_log:
        states = game[0]
        actions = game[1]
        rewards = game[2]
        if epoch == 0:
            move_avs = game[4][0]
            avs = game[4][1]
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

        # iterate over steps of the game
        for i in range(len(sarsa_values)):

            # keep track of the moment we go off-policy
            no_random_move_so_far = True

            # iterate over steps j after step i
            for j in range(len(sarsa_values) - i - 1):

                # get discounted rewards on future steps
                discounted_rewards = [gamma**u * rewards[i + u] for u in range(j + 1)]

                # if not cutting at random move
                if no_random_move_so_far or not cutting:
                    sarsa_values[i] = sarsa_values[i] \
                        + (1 - lambd) * (lambd**j) * ((gamma)**(j + 1) * avs[i + j + 1].detach().cpu().numpy()
                                                      + sum(discounted_rewards[0:]))

                    # for policy cutting
                    if random_moves[i + j + 1]:
                        final_index = j
                        no_random_move_so_far = False

                # ignore this (only useful when policy cutting)
                elif cutting and (not no_random_move_so_far):
                    j_ = final_index
                    discounted_rewards = [gamma**u * rewards[i + u] for u in range(j_ + 1)]
                    sarsa_values[i] = sarsa_values[i] \
                        + (1 - lambd) * (lambd**j) * ((gamma)**(j_ + 1) * avs[i + j_ + 1].detach().cpu().numpy() 
                                                      + sum(discounted_rewards))

            # do the same for index i
            if no_random_move_so_far or not cutting:
                discounted_rewards = [gamma**u * rewards[i + u] for u in range(len(sarsa_values) - i)]
                sarsa_values[i] = sarsa_values[i] + (lambd) ** (len(sarsa_values) - i - 1) * sum(discounted_rewards)

            # ignore
            else:
                j_ = final_index
                discounted_rewards = [gamma**u * rewards[i + u] for u in range(j_ + 1)]
                sarsa_values[i] = sarsa_values[i] \
                    + (lambd) ** (len(sarsa_values) - i - 1) * ((gamma)**j_ * avs[i + j_ + 1].detach().cpu().numpy() 
                                                                + sum(discounted_rewards))

        # calculate advantage and loss
        if len(move_avs) > 0:
            advantage = torch.from_numpy(sarsa_values).to(device) - torch.cat(move_avs).reshape(sarsa_values.shape)
            loss = loss + torch.pow(advantage, 2).mean()
            advantages.append(advantage)
        if printing:
            print('iteration')
            print('rewards', rewards)
            print('sarsa', sarsa_values)
            print('random', random_moves)
            print('avs', avs)
            print('move avs', move_avs)
            print('td errors', torch.cat(move_avs))

    if return_advantage:
        return loss, advantages

    return loss
