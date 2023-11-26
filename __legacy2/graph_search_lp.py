import numpy as np
from parameters import Parameters
import networkx as nx
from matplotlib import pyplot as plt
import copy
import cvxpy as cp

parameters = Parameters()
BOARD_SIZE = parameters.board_size


class BoardGraph:
    def __init__(self):

        """
        A stored graph representing all routes possible in the game, designed for fast searching over possible moves
        """

        # create the board graph
        self.boards = [nx.DiGraph(), nx.DiGraph()]

        # add all routes to the board
        self.boards[0].add_nodes_from([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])
        self.boards[0].add_edges_from([((i, j), (i + 1, j)) for i in range(BOARD_SIZE - 1) for j in range(BOARD_SIZE)])
        self.boards[0].add_edges_from([((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1) for j in range(BOARD_SIZE)])
        self.boards[0].add_edges_from([((i, j + 1), (i, j)) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE - 1)])
        self.boards[0].add_edges_from([((i, j), (i, j + 1)) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE - 1)])
        self.boards[1] = self.boards[0].copy()
        self.boards[1].add_edges_from([((0, j), (-1, BOARD_SIZE // 2)) for i in range(BOARD_SIZE)
                                       for j in range(BOARD_SIZE)])
        self.boards[1].add_edges_from([((-1, BOARD_SIZE // 2), (0, j)) for i in range(BOARD_SIZE)
                                       for j in range(BOARD_SIZE)])
        self.boards[0].add_edges_from([((BOARD_SIZE - 1, j), (BOARD_SIZE, BOARD_SIZE // 2))
                                       for j in range(BOARD_SIZE)])
        self.boards[0].add_edges_from([((BOARD_SIZE, BOARD_SIZE // 2), (BOARD_SIZE - 1, j))
                                       for j in range(BOARD_SIZE)])

        self.As = [nx.incidence_matrix(self.boards[0], oriented=True).todense(),
                   nx.incidence_matrix(self.boards[1], oriented=True).todense()]
        self.cut_As = [np.zeros((self.As[0].shape[1], self.As[0].shape[1])), np.zeros((self.As[0].shape[1],
                                                                                       self.As[0].shape[1]))]
        self.cut_b = np.zeros((self.As[0].shape[1], 1))

        self.last_y1_solutions = [np.zeros((self.As[0].shape[0], 1)), np.zeros((self.As[0].shape[0], 1))]
        self.last_y2_solutions = [np.zeros((self.As[0].shape[1], 1)), np.zeros((self.As[0].shape[1], 1))]

        # set up cp problem
        self.y1 = cp.Variable((self.As[0].shape[0], 1))
        self.y2 = cp.Variable((self.cut_As[0].shape[0], 1))

        self.cut_A_parameter = cp.Parameter(self.cut_As[0].shape)
        self.A_parameter = cp.Parameter(self.As[0].shape)
        self.cut_b_parameter = cp.Parameter(self.cut_b.shape)
        self.b_parameter = cp.Parameter((self.As[0].shape[0], 1))
        self.c_parameter = cp.Parameter((self.cut_As[0].shape[1], 1))
        self.c_parameter.value = np.zeros((self.cut_As[0].shape[1], 1))
        self.constraints = [self.A_parameter.T @ self.y1 + self.cut_A_parameter.T @ self.y2 + self.c_parameter >= 0]
        self.objective = - (self.b_parameter.T @ self.y1 + self.cut_b_parameter.T @ self.y2)
        self.problem = cp.Problem(cp.Maximize(self.objective), self.constraints)

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

    def copy_graph(self, board_graph):

        """
        makes a copy of the graphs stored in this class onto another class

        inputs:
            board_graph: Graph
                the other graph
        """
        self.boards = [board_graph.boards[0].copy(), board_graph.boards[1].copy()]
        self.As = copy.deepcopy(board_graph.As)
        self.cut_As = copy.deepcopy(board_graph.cut_As)
        self.cut_b = copy.copy(board_graph.cut_b)

        self.last_y1_solutions = copy.deepcopy(board_graph.last_y1_solutions)
        self.last_y2_solutions = copy.deepcopy(board_graph.last_y2_solutions)

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
            vertices = [v.astype('int') for v in vertices]
            edges = [(tuple(vertex), tuple(vertex + np.array([1, 0]))) for vertex in vertices]
        else:
            vertices = [wall_pos_command, wall_pos_command + np.array([1, 0])]
            vertices = [v.astype('int') for v in vertices]
            edges = [(tuple(vertex), tuple(vertex + np.array([0, 1]))) for vertex in vertices]

        edges = [(e[0], e[1]) for e in edges] + [(e[1], e[0]) for e in edges]
        all_edges_0 = list(self.boards[0].edges())
        all_edges_1 = list(self.boards[1].edges())

        for edge in edges:
            index = all_edges_0.index(edge)
            self.cut_As[0][index, index] = 1
            index = all_edges_1.index(edge)
            self.cut_As[1][index, index] = 1
        return None

    def check_both_players_can_reach_end(self, player_1_loc, player_2_loc, return_player_move=None):

        """
        checks that in the current state both players can make it to their goal
        """

        player_1_loc = player_1_loc.flatten()
        player_2_loc = player_2_loc.flatten()

        nodes_p1 = list(self.boards[0].nodes())
        nodes_p2 = list(self.boards[1].nodes())

        p1_node = (int(player_1_loc[0]), int(player_1_loc[1]))
        p2_node = (int(player_2_loc[0]), int(player_2_loc[1]))

        end_pos_p1 = nodes_p1.index((BOARD_SIZE, BOARD_SIZE // 2))
        end_pos_p2 = nodes_p2.index((-1, BOARD_SIZE // 2))
        pos_p1 = nodes_p1.index(p1_node)
        pos_p2 = nodes_p2.index(p2_node)

        pos_tuples = [(pos_p1, end_pos_p1), (pos_p2, end_pos_p2)]
        for i in range(2):
            (pos, end_pos) = pos_tuples[i]
            b = np.zeros((self.As[0].shape[0], 1))
            b[pos] = 1
            b[end_pos] = -1

            self.b_parameter.value = b
            self.cut_b_parameter.value = self.cut_b
            self.cut_A_parameter.value = self.cut_As[i]
            self.A_parameter.value = self.As[i]

            self.y1.value = self.last_y1_solutions[i]
            self.y2.value = self.last_y2_solutions[i]

            if return_player_move == i:
                self.c_parameter = np.zeros((self.cut_As[0].shape[1], 1))
            else:
                self.c_parameter = np.ones((self.cut_As[0].shape[1], 1))

            self.problem.solve(solver='ECOS', warm_start=True)  # reoptimize=True)

            if return_player_move == i:
                return self.constraints[0].dual_value

            self.last_y1_solutions[i] = self.y1.value
            self.last_y2_solutions[i] = self.y2.value

            if self.problem.value == np.inf:
                return False

        return True

    def get_sp_move(self, p1_loc, p2_loc, player):

        route_vector = self.check_both_players_can_reach_end(p1_loc, p2_loc, player)

        locs = [p1_loc, p2_loc]
        loc = locs[player]
        loc = loc.flatten()

        p1_node = (int(loc[0]), int(loc[1]))

        x = route_vector
        pahts_used = np.where(x > 0.5)[0]
        p1_neighbours = self.boards[player].neighbors(p1_node)
        p1_edges = [(p1_node, n) for n in p1_neighbours]
        edges_p1 = list(self.boards[player].edges())
        p1_edgelocs = [edges_p1.index(n) for n in p1_edges]
        p1_move_loc = [e for e in p1_edgelocs if e in pahts_used]
        p1_move_edge = edges_p1[p1_move_loc]
        p1_move = np.array([p1_move_edge[1][0] - p1_move_edge[0][0], p1_move_edge[1][1] - p1_move_edge[0][1], 0, 0, 0,
                            0])
        return p1_move
