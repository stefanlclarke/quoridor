from templates.parallel_trainer import ParallelTrainer
from models.q_models import QNetBot
from models.actor_models import Actor
import time
import torch.multiprocessing as mp


if __name__ == '__main__':

    t0 = time.time()
    cpus = mp.cpu_count()
    critic = QNetBot('3x256_9x9_16Feb224', good=True)
    actor = Actor()
    trainer = ParallelTrainer(cpus, critic.net, actor, iterations_per_worker=2)

    trainer.train(1)
    t1 = time.time()

    print('time taken {}'.format(t1 - t0))
