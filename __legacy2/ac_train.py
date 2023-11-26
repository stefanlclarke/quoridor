from trainers.ac_trainer import ACTrainer
import time


critic_info = {'input_dim': 3**2 * 4 + 2 * (1 + 1), 'critic_size_hidden': 128, 'critic_num_hidden': 2}
actor_info = {'input_dim': 3**2 * 4 + 2 * (1 + 1), 'actor_num_hidden': 2, 'actor_size_hidden': 128,
              'actor_output_dim': 4 + 2 * (3 - 1)**2, 'softmax_regularizer': 1}

trainer = ACTrainer(board_size=3, start_walls=1, critic_info=critic_info, actor_info=actor_info,
                    random_proportion=0.4, games_per_iter=100, decrease_epsilon_every=100,
                    qnet=None, actor=None, iterations_only_actor_train=0, convolutional=False, learning_rate=1e-4,
                    epsilon_decay=0.95, epsilon=0.4, minimum_epsilon=0.05, entropy_constant=1, max_grad_norm=1e5)
get_time_info = True

t0 = time.time()
time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
    illegal_move_handling_time, checking_winner_time, wall_handling_time = trainer.train(100, 10, 'ac_25_Nov',
                                                                                         get_time_info=get_time_info,
                                                                                         print_every=10)
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
