from game.command_line_player import CommandLinePlayer
from models.q_models import QNetBot
from models.actor_models import Actor
from tree_search.tree_search_minimax import MMTreeSearch

net = QNetBot('3x128_9x9_4Feb66')
tree = MMTreeSearch(net.net, 2, 2)
game = CommandLinePlayer(agent_2=tree)
game.play()
