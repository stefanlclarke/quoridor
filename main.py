from trainers.qtrainer import QTrainer
from trainers.ac_trainer import ACTrainer
from models.q_models import QNetBot
from models.actor_models import Actor
import time

trainer = QTrainer()
get_time_info = False

t0 = time.time()
trainer.train(1000000000, 200, '3x256_9x9_16Feb', get_time_info=get_time_info)
t1 = time.time()

if get_time_info:
    total_time = t1 - t0
    print('total time {}'.format(total_time))
    print('time playing {}'.format(playing))
    print('time learning {}'.format(learning))
    print('game processing time {}'.format(game))
    print('on policy time {}'.format(onp))
    print('off policy time {}'.format(offp))
    print('game moving time {}'.format(moving))
    print('illegal move handling time {}'.format(illegaling))
    print('winner checking time {}'.format(checking))
    print('wall handling time {}'.format(wall_handling))
