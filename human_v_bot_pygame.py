from pygame_.pygame_player import PygamePlayer
from models.q_models import QNetBot
from models.actor_models import Actor
from tree_search.tree_search_minimax import MMTreeSearch
from tree_search.tree_search_ac import ACTreeSearch

actor = Actor()
net = QNetBot('2x128_5x5_8Feb309', good=True)
tree = MMTreeSearch(net.net, 3, 2, min_value=-0.5, check_shortest_path=True, controlling=1)

game = PygamePlayer(agent_1=tree)
game.play()
