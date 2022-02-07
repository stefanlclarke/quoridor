from game.command_line_player import CommandLinePlayer
from models.q_models import QNetBot
from models.actor_models import Actor
from tree_search.tree_search_minimax import MMTreeSearch
from tree_search.tree_search_ac import ACTreeSearch

actor = Actor()
net = QNetBot('3x128_9x9_4Feb66')
tree = ACTreeSearch(net.net, actor, 4, 4, max_prob_check=0.4)
game = CommandLinePlayer(agent_2=tree)
game.play()
