import sys
import hydra
import os

from trainers.ac_trainer import ACTrainer
from trainers.ac_parallel_trainer import ParallelACTrainer
from trainers.qtrainer import QTrainer
from config import config, update_config_from_yaml
import datetime


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_ac(cfg):

    input_dim = cfg['BOARD_SIZE']**2 * 4 + 2 * (cfg['NUMBER_OF_WALLS'] + 1)
    output_dim = 4 + 2 * (cfg['BOARD_SIZE'] - 1)**2

    critic_info = {'input_dim': input_dim,
                   'critic_size_hidden': cfg['CRITIC_SIZE_HIDDEN'],
                   'critic_num_hidden': cfg['CRITIC_NUM_HIDDEN']}

    actor_info = {'input_dim': input_dim,
                  'actor_num_hidden': cfg['ACTOR_NUM_HIDDEN'],
                  'actor_size_hidden': cfg['ATOR_SIZE_HIDDEN'],
                  'actor_output_dim': output_dim,
                  'softmax_regularizer': cfg['SOFTMAX_REGULARIZER']}

    trainer = ACTrainer()

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


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_ac_parallel(cfg):

    update_config_from_yaml(config, cfg)

    os.mkdir(sys.argv[1][14:] + '/saves')

    input_dim = config.BOARD_SIZE**2 * 4 + 2 * (config.NUMBER_OF_WALLS + 1)
    output_dim = 4 + 2 * (config.BOARD_SIZE - 1)**2

    critic_info = {'input_dim': input_dim,
                   'critic_size_hidden': config.CRITIC_SIZE_HIDDEN,
                   'critic_num_hidden': config.CRITIC_NUM_HIDDEN}

    actor_info = {'input_dim': input_dim,
                  'actor_num_hidden': config.ACTOR_NUM_HIDDEN,
                  'actor_size_hidden': config.ACTOR_SIZE_HIDDEN,
                  'actor_output_dim': output_dim,
                  'softmax_regularizer': config.SOFTMAX_REGULARIZER}

    trainer = ParallelACTrainer(critic_info, actor_info, save_name='save',
                                save_directory=sys.argv[1][14:] + '/')

    trainer.train(config.TOTAL_EPOCHS, print_every=config.PRINT_EVERY)


if __name__ == "__main__":
    base = 'hydra.run.dir=outputs'

    if sys.argv[1] == 'ac':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        train_ac()

    if sys.argv[1] == 'acp':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        train_ac_parallel()

    if sys.argv[1] == 'q':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        train_q()
