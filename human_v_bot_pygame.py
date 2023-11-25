from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import ActorBot
from game.shortest_path_lp import ShortestPathBot

#net = QNetBot('AC5by511Nov20000_critic')
bot = ShortestPathBot(1)
game = PygamePlayer(agent_1=bot)
game.play()
