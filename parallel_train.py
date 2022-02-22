from trainers.q_parallel_trainer import ParallelTrainer
from models.q_models import QNetBot
from models.actor_models import Actor
import time
import torch.multiprocessing as mp


if __name__ == '__main__':

    t0 = time.time()
    cpus = mp.cpu_count()
    print(cpus)
    critic = QNetBot('3x256_7x7_22Feb60')
    trainer = ParallelTrainer(cpus, critic.net, '3x256_7x7_22Feb', iterations_per_worker=4, save_freq=1)

    trainer.train(1)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
