from trainers.q_parallel_trainer import ParallelTrainer
from models.q_models import QNet, QNetConv
import time
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

    if convolutional:
        critic = QNetConv()
    else:
        critic = QNet()
    trainer = ParallelTrainer(cpus, critic, '3x256_5x5_8Nov_conv', iterations_per_worker=iterations_per_worker,
                              save_freq=save_freq, convolutional=convolutional)

    trainer.train(n_epochs)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
    print('num games played {}'.format(iterations_per_worker * cpus * games_per_backprop))
