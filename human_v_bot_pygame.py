from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import Actor
from tree_search.tree_search_minimax import MMTreeSearch
from tree_search.tree_search_ac import ACTreeSearch

actor = Actor()
net = QNetBot('3x256_9x9_16Feb224')
tree = MMTreeSearch(net.net, 3, 2, min_value=-0.5, check_shortest_path=True, controlling=1)

game = PygamePlayer(agent_1=net)
game.play()
