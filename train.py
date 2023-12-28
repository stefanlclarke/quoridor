import sys
import hydra
import os
import datetime
import torch

from trainers.ac_trainer import ACTrainer
from trainers.ac_parallel_trainer import ParallelACTrainer
from trainers.qtrainer import QTrainer
from config import config, update_config_from_yaml
from models.actor_models import Actor
from models.critic_models import Critic, CriticConv


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_ac(cfg):

    update_config_from_yaml(config, cfg)

    if cfg['CONTINUE'] is None:
        os.mkdir(sys.argv[1][14:] + '/saves')
    else:
        sys.argv[1] = 'hydra.run.dir=' + cfg['CONTINUE']

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

    if cfg['CONTINUE'] is None:
        trainer = ACTrainer(critic_info, actor_info, save_name='save',
                            save_directory=sys.argv[1][14:])
    else:
        central_actor = Actor(**actor_info)
        central_critic = Critic(**critic_info)
        trainer = ACTrainer(critic_info, actor_info, save_name='save',
                            save_directory=sys.argv[1][14:], central_actor=central_actor,
                            central_critic=central_critic)

    time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
        illegal_move_handling_time, checking_winner_time, wall_handling_time \
        = trainer.train(config.TOTAL_EPOCHS, config.SAVE_EVERY, 'AC', get_time_info=True,
                        print_every=config.PRINT_EVERY)


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_q(cfg):

    input_dim = config.BOARD_SIZE**2 * 4 + 2 * (config.NUMBER_OF_WALLS + 1)
    output_dim = 4 + 2 * (config.BOARD_SIZE - 1)**2

    qnet_parameters = {'input_dim': input_dim,
                       'hidden_size': config.CRITIC_SIZE_HIDDEN,
                       'num_hidden': config.CRITIC_NUM_HIDDEN,
                       'actor_output_dim': output_dim}

    trainer = QTrainer(qnet_parameters=qnet_parameters,
                       save_name='save',
                       save_directory=sys.argv[1][14:])

    time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
        illegal_move_handling_time, checking_winner_time, wall_handling_time \
        = trainer.train(config.TOTAL_EPOCHS, config.SAVE_EVERY, 'Q', get_time_info=True, print_every=10)


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='run.yaml')
def train_ac_parallel(cfg):

    if cfg['CONTINUE'] is None:
        os.mkdir(sys.argv[1][14:] + '/saves')
    else:
        sys.argv[1] = 'hydra.run.dir=' + cfg['CONTINUE']

    update_config_from_yaml(config, cfg)

    input_dim = config.BOARD_SIZE**2 * 4 + 2 * (config.NUMBER_OF_WALLS + 1)
    output_dim = 4 + 2 * (config.BOARD_SIZE - 1)**2

    if not cfg['USE_CONV_NET']:
        critic_info = {'input_dim': input_dim,
                       'critic_size_hidden': config.CRITIC_SIZE_HIDDEN,
                       'critic_num_hidden': config.CRITIC_NUM_HIDDEN}
    else:
        critic_info = {'input_dim': input_dim,
                       'critic_size_hidden': config.CRITIC_SIZE_HIDDEN,
                       'critic_num_hidden': config.CRITIC_NUM_HIDDEN,
                       'sidelen': config.CONV_SIDELEN,
                       'conv_internal_channels': config.CONV_INTERNAL_CHANNELS,
                       'linear_in_dim': config.NUMBER_OF_WALLS * 2 + 2,
                       'num_conv': config.CONV_NUM_LAYERS,
                       'kernel_size': config.CONV_KERNEL_SIZE
                       }

    actor_info = {'input_dim': input_dim,
                  'actor_num_hidden': config.ACTOR_NUM_HIDDEN,
                  'actor_size_hidden': config.ACTOR_SIZE_HIDDEN,
                  'actor_output_dim': output_dim,
                  'softmax_regularizer': config.SOFTMAX_REGULARIZER}

    if cfg['CONTINUE'] is None:
        trainer = ACTrainer(critic_info, actor_info, save_name='save',
                            save_directory=sys.argv[1][14:])
    else:
        actir_dict, critic_dict = load_old_models(sys.argv[1][14:] + '/saves/', 'save')

        central_actor = Actor(**actor_info)
        central_actor.load_state_dict(actir_dict)

        if not cfg['USE_CONV_NET']:
            central_critic = Critic(**critic_info)
        else:
            central_critic = CriticConv(**critic_info)
            central_critic.load_state_dict(critic_dict)
        trainer = ACTrainer(critic_info, actor_info, save_name='save',
                            save_directory=sys.argv[1][14:], central_actor=central_actor,
                            central_critic=central_critic)

    trainer = ParallelACTrainer(critic_info, actor_info, save_name='save',
                                save_directory=sys.argv[1][14:] + '/')

    trainer.train(config.TOTAL_EPOCHS, print_every=config.PRINT_EVERY)


def load_old_models(dir, save_name):

    old_models = os.listdir(dir)
    old_actors = [x for x in old_models if x[-5:] == 'ACTOR']
    prev_savenums = sorted([int(x[4:-5]) for x in old_actors])
    choice = prev_savenums[-1]

    return torch.load(dir + save_name + str(choice) + 'ACTOR'), (torch.load(dir + save_name + str(choice)))


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
