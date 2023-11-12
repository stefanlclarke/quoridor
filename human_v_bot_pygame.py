from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import ActorBot

net = QNetBot('AC5by511Nov20000_critic')
game = PygamePlayer(agent_1=net)
game.play()
