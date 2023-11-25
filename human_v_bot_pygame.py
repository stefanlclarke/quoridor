from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import ActorBot
from game.shortest_path_lp import ShortestPathBot

net = ActorBot('ac_25_nov110ACTOR')
game = PygamePlayer(agent_1=net)
game.play()
