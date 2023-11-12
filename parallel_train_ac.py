from trainers.ac_parallel_trainer import ParallelTrainer
from models.q_models import QNet, QNetConv, QNetBot
from models.actor_models import Actor
import time
import torch
from parameters import Parameters

parameters = Parameters()

iterations_per_worker = parameters.backprops_per_worker
games_per_backprop = parameters.games_between_backprops
save_freq = parameters.save_every
n_epochs = parameters.epochs
convolutional = parameters.convolutional
n_cpus = parameters.n_cores

if __name__ == '__main__':

    t0 = time.time()
    cpus = n_cpus
    print(f'I am using {cpus} CPUs')

    actor = Actor()
    if convolutional:
        critic = QNetConv()
    else:
        critic = QNet()
    # critic.load_state_dict(torch.load('saves/AC5by511Nov20000_critic'))
    # actor.load_state_dict(torch.load('saves/AC5by511Nov20000_actor'))

    trainer = ParallelTrainer(cpus, critic, actor, 'AC3by312Nov', convolutional=convolutional)

    trainer.train(n_epochs)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
