import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import numpy as np

from quoridor_ai.envs import get_env
from quoridor_ai.model import ActorCritic, ActorCriticAgent
from quoridor_ai.settings import settings


OPPONENT_SAVE_DIR = 'quoridor_ai/saves'


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None, max_steps=1):
    torch.manual_seed(args.seed + rank)

    env = get_env()
    # env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    for _ in range(max_steps):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, settings.lstm_dimension)
            hx = torch.zeros(1, settings.lstm_dimension)
        else:
            cx = cx.detach()
            hx = hx.detach()

        if _ % args.reload_every == 0:
            new_opponent = load_opponent()
            env.load_new_opponent(new_opponent)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy().flatten()[0])
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if time.time() - args.last_save > args.seconds_per_save and rank == 0:
            passed_time = time.time() - args.init_time
            shared_model.save(str(passed_time))
            args.last_save = time.time()
            print_iteration(str(passed_time), str(policy_loss.flatten().detach().numpy()[0]),
                            str(value_loss.flatten().detach().numpy()[0]))


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)


def load_opponent():
    """
    Chooses an old version of self and loads it in as the opponent
    """

    old_models = os.listdir(OPPONENT_SAVE_DIR)

    if len(old_models) == 0:
        opponent = ActorCriticAgent(None, get_env())
        return opponent

    prev_savenums = sorted([int(float(x)) for x in old_models])
    acceptable_choices = prev_savenums[-settings.load_from_last:]

    if settings.load_distribution == "uniform":
        choice = np.random.choice(acceptable_choices)
    elif settings.load_distribution == "geometric":
        choice_ind = np.random.geometric(p=0.5)
        if choice_ind >= len(acceptable_choices):
            choice_ind = len(acceptable_choices)
        choice = old_models[-choice_ind]
    else:
        raise ValueError("invalid distribution")

    opponent = ActorCriticAgent(OPPONENT_SAVE_DIR + '/' + str(choice), get_env())
    return opponent
