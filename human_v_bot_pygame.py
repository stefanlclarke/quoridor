from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot

net = QNetBot('3x256_5x5_7Nov360')
game = PygamePlayer(agent_1=net)
game.play()
