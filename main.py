
from trainers.ac_trainer import ACTrainer
import time

trainer = ACTrainer()
get_time_info = True

t0 = time.time()
time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
    illegal_move_handling_time, checking_winner_time, wall_handling_time \
    = trainer.train(16, 16, 'ACtest', get_time_info=get_time_info)
t1 = time.time()

if get_time_info:
    total_time = t1 - t0
    print('total time {}'.format(total_time))
    print('time playing {}'.format(time_playing))
    print('time learning {}'.format(time_learning))
    print('game processing time {}'.format(game_processing_time))
    print('on policy time {}'.format(on_policy_time))
    print('off policy time {}'.format(off_policy_time))
    print('game moving time {}'.format(moving_time))
    print('illegal move handling time {}'.format(illegal_move_handling_time))
    print('winner checking time {}'.format(checking_winner_time))
    print('wall handling time {}'.format(wall_handling_time))
