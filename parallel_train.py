from trainers.q_parallel_trainer import ParallelTrainer
from models.q_models import QNet, QNetConv
import time
import torch.multiprocessing as mp

iterations_per_worker = 4
save_freq = 100
n_epochs = 1000000
convolutional = False

if __name__ == '__main__':

    t0 = time.time()
    cpus = mp.cpu_count()
    print(f'I am using {cpus} CPUs')

    if convolutional:
        critic = QNetConv('3x256_5x5_7Nov_conv14670')
    else:
        critic = QNet()
    trainer = ParallelTrainer(cpus, critic, '3x256_5x5_8Nov_conv', iterations_per_worker=iterations_per_worker,
                              save_freq=save_freq, convolutional=convolutional)

    trainer.train(n_epochs)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
    print('num games played {}'.format(iterations_per_worker * cpus))
