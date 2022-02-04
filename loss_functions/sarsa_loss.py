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
cutting = parameters.cut_at_random_move

def sarsa_loss(memory, net, epoch, possible_moves, printing=False, return_advantage=False):
    loss = torch.tensor([0.]).to(device)
    advantages = []
    for game in memory.game_log:
        #print(game)
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
                move_value = values[actions[i]]
                move_avs.append(move_value.unsqueeze(0))
                avs.append(best_move_value.unsqueeze(0))
                random_moves.append(True)

        sarsa_values = np.zeros(len(rewards))
        for i in range(len(sarsa_values)):
            no_random_move_so_far = True
            for j in range(len(sarsa_values) - i - 1):
                discounted_rewards = [gamma**u * rewards[i+u] for u in range(j+1)]
                if no_random_move_so_far or not cutting:
                    sarsa_values[i] = sarsa_values[i] + (1 - lambd) * (lambd**j) * ((gamma)**(j+1) * avs[i+j+1].detach().cpu().numpy() + sum(discounted_rewards))
                    if random_moves[i+j+1]:
                        final_index = j
                        no_random_move_so_far = False
                elif cutting and (not no_random_move_so_far):
                    j_ = final_index
                    discounted_rewards = [gamma**u * rewards[i+u] for u in range(j_+1)]
                    sarsa_values[i] = sarsa_values[i] +  (1 - lambd) * (lambd**j) * ((gamma)**(j_+1) * avs[i+j_+1].detach().cpu().numpy() + sum(discounted_rewards))
                #no_random_move_so_far = no_random_move_so_far and not random_moves[i+j]
            if no_random_move_so_far or not cutting:
                discounted_rewards = [gamma**u * rewards[i+u] for u in range(len(sarsa_values) - i)]
                sarsa_values[i] = sarsa_values[i] + (lambd) **(len(sarsa_values)- i - 1) * sum(discounted_rewards)
            else:
                j_ = final_index
                discounted_rewards = [gamma**u * rewards[i+u] for u in range(j_+1)]
                sarsa_values[i] = sarsa_values[i] +  (lambd) **(len(sarsa_values)- i - 1) * ((gamma)**j_ * avs[i+j_+1].detach().cpu().numpy() + sum(discounted_rewards))
        if len(move_avs) > 0:
            advantage = torch.cat(move_avs) - torch.from_numpy(sarsa_values).to(device)
            loss = loss + torch.pow(advantage, 2).mean()
            advantages.append(-advantage)
        if printing:
            print('iteration')
            print('rewards', rewards)
            print('sarsa', sarsa_values)
            print('random', random_moves)
            print('avs', avs)
            print('move avs', move_avs)
            print('td errors', torch.cat(move_avs))
            print('loss contrbution', torch.pow(torch.cat(move_avs), 2).mean())
    #print('FINAL LOSS', loss)

    if return_advantage:
        return loss, advantages

    return loss
