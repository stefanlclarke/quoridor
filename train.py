import sys
import hydra

from trainers.ac_trainer import ACTrainer
from trainers.qtrainer import QTrainer
import datetime


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_ac(cfg):

    input_dim = cfg['board_size']**2 * 4 + 2 * (cfg['number_of_walls'] + 1)
    output_dim = 4 + 2 * (cfg['board_size'] - 1)**2

    critic_info = {'input_dim': input_dim,
                   'critic_size_hidden': cfg['critic_size_hidden'],
                   'critic_num_hidden': cfg['critic_num_hidden']}

    actor_info = {'input_dim': input_dim,
                  'actor_num_hidden': cfg['actor_num_hidden'],
                  'actor_size_hidden': cfg['actor_size_hidden'],
                  'actor_output_dim': output_dim,
                  'softmax_regularizer': cfg['softmax_regularizer']}

    trainer = ACTrainer(board_size=cfg['board_size'],
                        start_walls=cfg['number_of_walls'],
                        critic_info=critic_info,
                        actor_info=actor_info,
                        random_proportion=cfg['random_proportion'],
                        games_per_iter=cfg['games_between_backprops'],
                        decrease_epsilon_every=cfg['decrease_epsilon_every'],
                        qnet=None,
                        actor=None,
                        iterations_only_actor_train=cfg['iterations_only_actor_train'],
                        convolutional=cfg['convolutional'],
                        learning_rate=cfg['learning_rate'],
                        epsilon_decay=cfg['epsilon_decay'],
                        epsilon=cfg['epsilon'],
                        minimum_epsilon=cfg['minimum_epsilon'],
                        entropy_constant=cfg['entropy_constant'],
                        max_grad_norm=cfg['max_grad_norm'],
                        save_name=sys.argv[1][14:] + '/save',
                        lambd=cfg['lambd'],
                        gamma=cfg['gamma'],
                        move_prob=cfg['move_prob'],
                        minimum_move_prob=cfg['minimum_move_prob'],
                        entropy_bias=cfg['entropy_bias'],
                        total_reset_every=cfg['total_reset_every'])

    time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
        illegal_move_handling_time, checking_winner_time, wall_handling_time \
        = trainer.train(cfg['epochs'], cfg['save_every'], 'AC', get_time_info=True)
    

@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_q(cfg):

    input_dim = cfg['board_size']**2 * 4 + 2 * (cfg['number_of_walls'] + 1)
    output_dim = 4 + 2 * (cfg['board_size'] - 1)**2

    qnet_parameters = {'input_dim': input_dim,
                       'hidden_size': cfg['critic_size_hidden'],
                       'num_hidden': cfg['critic_num_hidden'],
                       'actor_output_dim': output_dim}

    trainer = QTrainer(board_size=cfg['board_size'],
                       start_walls=cfg['number_of_walls'],
                       qnet_parameters=qnet_parameters,
                       random_proportion=cfg['random_proportion'],
                       games_per_iter=cfg['games_between_backprops'],
                       decrease_epsilon_every=cfg['decrease_epsilon_every'],
                       net=None,
                       learning_rate=cfg['learning_rate'],
                       epsilon_decay=cfg['epsilon_decay'],
                       epsilon=cfg['epsilon'],
                       minimum_epsilon=cfg['minimum_epsilon'],
                       save_name=sys.argv[1][14:] + '/save',
                       lambd=cfg['lambd'],
                       gamma=cfg['gamma'],
                       minimum_move_prob=cfg['minimum_move_prob'],
                       total_reset_every=cfg['total_reset_every'])

    time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
        illegal_move_handling_time, checking_winner_time, wall_handling_time \
        = trainer.train(cfg['epochs'], cfg['save_every'], 'Q', get_time_info=True, print_every=10)


if __name__ == "__main__":
    base = 'hydra.run.dir=outputs'

    if sys.argv[1] == 'ac':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        train_ac()

    if sys.argv[1] == 'q':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        train_q()
