from game.command_line_player import CommandLinePlayer
from models.q_models import QNetBot

bot = QNetBot('3x256_9x9_01Feb140')
game = CommandLinePlayer(agent_2=bot)
game.play()
