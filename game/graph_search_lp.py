import numpy as np
from parameters import Parameters
import networkx as nx
from matplotlib import pyplot as plt
from copy import copy
import cvxpy as cp

parameters = Parameters()
BOARD_SIZE = parameters.board_size


class BoardGraph:
    def __init__(self):

        """
        A stored graph representing all routes possible in the game, designed for fast searching over possible moves
        """

        # create the board graph
        self.board = nx.DiGraph()

        # add all routes to the board
        self.board.add_nodes_from([(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((i, j), (i + 1, j)) for i in range(BOARD_SIZE - 1) for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((i + 1, j), (i, j)) for i in range(BOARD_SIZE - 1) for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((i, j + 1), (i, j)) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE - 1)])
        self.board.add_edges_from([((0, j), (-1, BOARD_SIZE // 2)) for i in range(BOARD_SIZE)
                                   for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((-1, BOARD_SIZE // 2), (0, j)) for i in range(BOARD_SIZE)
                                   for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((BOARD_SIZE - 1, j), (BOARD_SIZE, BOARD_SIZE // 2))
                                   for j in range(BOARD_SIZE)])
        self.board.add_edges_from([((BOARD_SIZE, BOARD_SIZE // 2), (BOARD_SIZE - 1, j))
                                   for j in range(BOARD_SIZE)])

        self.A = nx.incidence_matrix(self.board)
        self.cut_edges = []
        self.cut_A = np.zeros((self.A.shape[1], self.A.shape[1]))
        self.cut_b = np.zeros((self.A.shape[1], 1))

        self.last_y1_solution = np.zeros((self.A.shape[0], 1))
        self.last_y2_solution = np.zeros((self.A.shape[1], 1))

        # set up cp problem
        self.y1 = cp.Variable((self.A.shape[0], 1))
        self.y2 = cp.Variable((self.cut_A.shape[0], 1))

        self.cut_A_parameter = cp.Parameter(self.cut_A.shape)
        self.cut_b_parameter = cp.Parameter(self.cut_b.shape)
        self.b_parameter = cp.Variable((self.A.shape[0], 1))
        self.constraints = [self.A.T @ self.y1 + self.cut_A_parameter.T @ self.y2 >= 0]
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
        self.board = board_graph.board.copy()
        self.last_dual_solution = copy.copy(board_graph.last_dual_solution)
        self.A = copy.copy(board_graph.A)
        self.cut_edges = copy.deepcopy(board_graph.cut_edges)

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

        edges = edges + [(e[1], e[0]) for e in edges]
        all_edges = list(self.board.edges())

        for edge in edges:
            index = all_edges.index(edge)
            self.cut_A[index, index] = 1
        return None

    def check_both_players_can_reach_end(self, player_1_loc, player_2_loc):

        """
        checks that in the current state both players can make it to their goal
        """

        player_1_loc = player_1_loc.flatten()
        player_2_loc = player_2_loc.flatten()

        nodes = list(self.board.nodes())

        p1_node = (int(player_1_loc[0]), int(player_1_loc[1]))
        p2_node = (int(player_2_loc[0]), int(player_2_loc[1]))

        end_pos_p1 = nodes.index((BOARD_SIZE, BOARD_SIZE // 2))
        end_pos_p2 = nodes.index((-1, BOARD_SIZE // 2))
        pos_p1 = nodes.index(p1_node)
        pos_p2 = nodes.index(p2_node)

        for (pos, end_pos) in [(pos_p1, end_pos_p1), (pos_p2, end_pos_p2)]:
            b = np.zeros((self.A.shape[0], 1))
            b[pos] = 1
            b[end_pos] = -1

            self.b_parameter.value = b
            self.cut_b_parameter.value = self.cut_b
            self.cut_A_parameter.value = self.cut_A

            self.y1.value = self.last_y1_solution
            self.y2.value = self.last_y2_solution

            self.problem.solve(solver='GUROBI', warm_start=True)

            if self.problem.value == np.inf:
                return False

        return True





        
