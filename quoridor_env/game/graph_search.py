import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import copy


class BoardGraph:
    def __init__(self, board_size):

        """
        A stored graph representing all routes possible in the game, designed for fast searching over possible moves
        """

        # create the graphs(board, player 1, player 2)
        self.board = nx.Graph()
        self.graph = [nx.Graph(), nx.Graph()]

        self.board_size = board_size

        # add all routes to the graph
        for k in range(2):
            self.graph[k].add_nodes_from([(i, j) for i in range(self.board_size) for j in range(self.board_size)])
            self.graph[k].add_edges_from([((i, j), (i + 1, j)) for i in range(self.board_size - 1)
                                          for j in range(self.board_size)])
            self.graph[k].add_edges_from([((i, j), (i, j + 1)) for i in range(self.board_size)
                                          for j in range(self.board_size - 1)])
            if k == 1:
                self.graph[k].add_nodes_from([(-1, self.board_size // 2)])
                self.graph[k].add_edges_from([((-1, self.board_size // 2), (0, j)) for j in range(self.board_size)])
            else:
                self.graph[k].add_nodes_from([(self.board_size, self.board_size // 2)])
                self.graph[k].add_edges_from([((self.board_size, self.board_size // 2), (self.board_size - 1, j))
                                              for j in range(self.board_size)])

        # add all routes to the board
        self.board.add_nodes_from([(i, j) for i in range(self.board_size) for j in range(self.board_size)])
        self.board.add_edges_from([((i, j), (i + 1, j)) for i in range(self.board_size - 1) for j in
                                   range(self.board_size)])
        self.board.add_edges_from([((i, j), (i, j + 1)) for i in range(self.board_size)
                                   for j in range(self.board_size - 1)])

        # stores best move at any given time
        self.direction_graph = [nx.DiGraph(), nx.DiGraph()]

        # add all routes to direction graphs
        for k in range(2):
            self.direction_graph[k].add_nodes_from(self.graph[k])
            if k == 0:
                self.direction_graph[k].add_edges_from([((i, j), (i + 1, j)) for j in range(self.board_size)
                                                        for i in range(self.board_size - 1)])
                self.direction_graph[k].add_edges_from([((self.board_size - 1, j),
                                                         (self.board_size, self.board_size // 2))
                                                        for j in range(self.board_size)])
            else:
                self.direction_graph[k].add_edges_from([((i + 1, j), (i, j)) for j in range(self.board_size)
                                                        for i in range(self.board_size - 1)])
                self.direction_graph[k].add_edges_from([((0, j), (-1, self.board_size // 2)) for i in
                                                        range(self.board_size)
                                                        for j in range(self.board_size)])

        # initialize paths on the direction graphs
        self.initialize_paths()

    def initialize_paths(self):

        """
        Initializes best paths on the current board
        """
        self.path_lengths = [nx.shortest_path_length(self.direction_graph[0],
                                                     target=(self.board_size, self.board_size // 2),
                                                     method='dijkstra'),
                             nx.shortest_path_length(self.direction_graph[1], target=(-1, self.board_size // 2),
                                                     method='dijkstra')]
        self.shortest_paths = [nx.shortest_path(self.direction_graph[0],
                                                target=(self.board_size, self.board_size // 2), method='dijkstra'),
                               nx.shortest_path(self.direction_graph[1], target=(-1, self.board_size // 2),
                                                method='dijkstra')]
        self.dependent_edges = [{}, {}]
        for k in range(2):
            for vertex in self.direction_graph[k].nodes:
                shortest_path = self.shortest_paths[k][vertex]
                node_dependent_edges = [(shortest_path[i], shortest_path[i + 1])
                                        for i in range(len(shortest_path) - 1)]
                self.dependent_edges[k][vertex] = set([frozenset(x) for x in node_dependent_edges])
        self.dependent_vertices = [{}, {}]
        for k in range(2):
            for edge in self.graph[k].edges:
                dependent_vertices = []
                for vertex in self.direction_graph[k].nodes:
                    if frozenset(edge) in self.dependent_edges[k][vertex]:
                        dependent_vertices.append(vertex)
                self.dependent_vertices[k][frozenset(edge)] = set(dependent_vertices)

    def plot(self):

        """
        plot function for visualizing the board graph
        """

        plt.figure(figsize=(5, 5))
        pos = {(x, y): (y, -x) for x, y in self.board.nodes()}
        nx.draw(self.board, pos=pos,
                node_color='lightgreen',
                with_labels=True,
                node_size=600)
        plt.show()

    def player_plot(self, player):

        """
        plot function for visualizing each player graph
        """
        graph = self.direction_graph[player]
        plt.figure(figsize=(5, 5))
        pos = {(x, y): (y, -x) for x, y in graph}
        nx.draw(graph, pos=pos,
                node_color='lightgreen',
                with_labels=True,
                node_size=600)
        plt.show()

    def reconfigure_paths(self, edges_to_cut):

        """
        given a list of edges to cut (because a wall was put down) reconfigure the graphs and work out
        best shortest paths

        inputs:
            edges_to_cut: list
                list of edges which should be cut
        """

        # delete edges on board
        self.board.remove_edges_from(edges_to_cut)

        # delete edges from each player graph
        for k in range(2):
            graph = self.graph[k]
            path_lengths = self.path_lengths[k]
            dependent_vertices = self.dependent_vertices[k]
            direction_graph = self.direction_graph[k]
            dependent_edges = self.dependent_edges[k]
            graph.remove_edges_from(edges_to_cut)
            for edge in edges_to_cut:
                direction_graph.remove_edges_from([edge])
                direction_graph.remove_edges_from([tuple(reversed(edge))])
            vertices_to_reconfigure = []
            neighbours_of_cut_vertices = []

            # update minimum route lengths
            for edge in edges_to_cut:
                vertices_to_reconfigure += list(dependent_vertices[frozenset(edge)])
            for vertex in vertices_to_reconfigure:
                path_lengths[vertex] = 2 * self.board_size**2
                neighbours_of_cut_vertices += graph[vertex]
                vertex_edges = direction_graph[vertex]
                direction_graph.remove_edges_from([(vertex, x) for x in vertex_edges])
                edges_no_longer_depending = dependent_edges[vertex]
                for edge in edges_no_longer_depending:
                    try:
                        dependent_vertices[edge].remove(vertex)
                    except KeyError:
                        print('edge', edge)
                        print('vertex', vertex)
                        print(dependent_vertices)
                        self.player_plot(k)
                        raise KeyError
                dependent_edges[vertex] = []

            current_active_set = neighbours_of_cut_vertices
            running = True
            while running:

                # once everything is checked break
                if len(current_active_set) == 0:
                    running = False
                    break

                # go over vertices checking neighbours and updating minimum length based on this
                current_active_set = sorted(current_active_set, key=lambda x: path_lengths[x])
                vertex_to_check = current_active_set[0]
                value = path_lengths[vertex_to_check]
                neighbours_of_vertex_to_check = graph[vertex_to_check]
                for vertex in neighbours_of_vertex_to_check:
                    current_value = path_lengths[vertex]
                    path_lengths[vertex] = min(value + 1, current_value)
                    if current_value > value + 1:
                        current_active_set.append(vertex)
                        direction_graph.add_edges_from([(vertex, vertex_to_check)])
                        dependent_edges[vertex] = set.union(set(dependent_edges[vertex_to_check]),
                                                            dependent_edges[vertex])
                        dependent_edges[vertex].add(frozenset((vertex, vertex_to_check)))
                current_active_set.remove(vertex_to_check)

            for vertex in vertices_to_reconfigure:
                for edge in dependent_edges[vertex]:
                    try:
                        dependent_vertices[edge].add(vertex)
                    except KeyError:
                        dependent_vertices[edge] = set([vertex])

    def copy_graph(self, board_graph):

        """
        makes a copy of the graphs stored in this class onto another class

        inputs:
            board_graph: Graph
                the other graph
        """
        self.board = board_graph.board.copy()
        self.graph = [g.copy() for g in board_graph.graph]
        self.direction_graph = [g.copy() for g in board_graph.direction_graph]
        self.path_lengths = copy.deepcopy(board_graph.path_lengths)
        self.shortest_paths = copy.deepcopy(board_graph.shortest_paths)
        self.dependent_edges = copy.deepcopy(board_graph.dependent_edges)
        self.dependent_vertices = copy.deepcopy(board_graph.dependent_vertices)

    def wall_placement(self, wall_pos_command, wall_orientation_command):

        """
        handles the placement of a wall

        inputs: 
            wall_pos_command: np.ndarray
                the position of the wall being placed

            wall_orientation_command: int
                the orientation of the wall being placed
        """
        if wall_orientation_command[1] == 1:
            vertices = [wall_pos_command, wall_pos_command + np.array([0, 1])]
            edges = [(tuple(vertex), tuple(vertex + np.array([1, 0]))) for vertex in vertices]
        else:
            vertices = [wall_pos_command, wall_pos_command + np.array([1, 0])]
            edges = [(tuple(vertex), tuple(vertex + np.array([0, 1]))) for vertex in vertices]
        self.reconfigure_paths(edges)
        return None

    def check_both_players_can_reach_end(self, player_1_loc, player_2_loc):

        """
        checks that in the current state both players can make it to their goal
        """
        if len(self.dependent_edges[0][tuple(player_1_loc)]) > 0:
            if len(self.dependent_edges[1][tuple(player_2_loc)]) > 0:
                return True
        return False
