from game.graph_search import BoardGraph
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

graph = BoardGraph()
graph.reconfigure_paths([((1,1),(2,1)),((1,2),(2,2))])
graph.player_plot(1)
graph.player_plot(0)
print(graph.check_both_players_can_reach_end(np.array([0,4]), np.array([4,0])))

print(nx.adjacency_matrix(graph.board).todense().shape)
