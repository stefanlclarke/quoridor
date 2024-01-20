import time
from dataclasses import dataclass


@dataclass
class Args:
    lr = 0.0005
    seed = 1
    gamma = 0.99
    gae_lambda = 1.
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 50
    num_processes = 4
    num_steps = 200
    max_episode_length = 200
    no_shared = False
    seconds_per_save = 1
    init_time = time.time()
    last_save = time.time()
    steps_per_worker = int(1e20)
    hidden_dimension = 512
    num_hidden = 3
    load_from_last = 200
    load_distribution = "geometric"
    reload_every = 1
    lstm_dimension = hidden_dimension


settings = Args()
