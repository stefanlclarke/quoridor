import hydra
import yaml
import sys
import datetime

from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import ActorBot
from config import config, update_config_from_yaml


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='play.yaml')
def play_actor(cfg):

    dir = cfg['save_dir']
    bot_dir = dir + '/' + cfg['save_name']
    bot_cfg_dir = dir + '/.hydra/config.yaml'

    with open(bot_cfg_dir, 'r') as stream:
        cfg = yaml.safe_load(stream)

    input_dim = cfg['board_size']**2 * 4 + 2 * (cfg['number_of_walls'] + 1)
    output_dim = 4 + 2 * (cfg['board_size'] - 1)**2

    net = ActorBot(bot_dir, cfg['actor_num_hidden'], cfg['actor_size_hidden'], input_dim, output_dim,
                   cfg['softmax_regularizer'])
    game = PygamePlayer(cfg['board_size'], cfg['number_of_walls'], agent_1=net)
    game.play()


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='play.yaml')
def play_q(cfg):

    dir = cfg['save_dir']
    bot_dir = dir + '/' + cfg['save_name']
    bot_cfg_dir = dir + '/.hydra/config.yaml'

    with open(bot_cfg_dir, 'r') as stream:
        cfg = yaml.safe_load(stream)

    input_dim = cfg['board_size']**2 * 4 + 2 * (cfg['number_of_walls'] + 1)
    output_dim = 4 + 2 * (cfg['board_size'] - 1)**2

    qnet_parameters = {'input_dim': input_dim,
                       'hidden_size': cfg['critic_size_hidden'],
                       'num_hidden': cfg['critic_num_hidden'],
                       'actor_output_dim': output_dim}

    net = QNetBot(bot_dir, qnet_parameters)
    game = PygamePlayer(cfg['board_size'], cfg['number_of_walls'], agent_1=net)
    game.play()


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='play.yaml')
def play_human(cfg):

    dir = cfg['save_dir']
    bot_cfg_dir = dir + '/.hydra/config.yaml'

    with open(bot_cfg_dir, 'r') as stream:
        cfg = yaml.safe_load(stream)

    update_config_from_yaml(config, cfg)

    game = PygamePlayer()
    game.play()


if __name__ == "__main__":
    base = 'hydra.run.dir=outputs'

    if sys.argv[1] == 'ac':
        sys.argv[1] = base + '/' + sys.argv[1] + 'play' + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        play_actor()

    if sys.argv[1] == 'q':
        sys.argv[1] = base + '/' + sys.argv[1] + 'play' + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        play_q()

    if sys.argv[1] == 'human':
        sys.argv[1] = base + '/' + sys.argv[1] + 'play' + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        play_human()
