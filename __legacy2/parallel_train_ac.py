from trainers.ac_parallel_trainer import ParallelTrainer
from models.critic_models import CriticConv, Critic
from models.actor_models import Actor
import time
from parameters import Parameters

parameters = Parameters()

iterations_per_worker = parameters.backprops_per_worker
games_per_backprop = parameters.games_between_backprops
save_freq = parameters.save_every
n_epochs = parameters.epochs
convolutional = parameters.convolutional
n_cpus = parameters.n_cores
zero_critic = parameters.zero_critic

if __name__ == '__main__':

    t0 = time.time()
    cpus = n_cpus
    print(f'I am using {cpus} CPUs')

    actor = Actor()
    if convolutional:
        critic = CriticConv()
    else:
        critic = Critic()

    if zero_critic:
        critic.zero_all_parameters()
    # critic.load_state_dict(torch.load('saves/AC5by511Nov20000_critic'))
    # actor.load_state_dict(torch.load('saves/AC5by511Nov20000_actor'))

    trainer = ParallelTrainer(cpus, critic, actor, 'AC3by312Nov', convolutional=convolutional)

    trainer.train(n_epochs)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
