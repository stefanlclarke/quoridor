import numpy as np
import copy
import time
from game.game_helper_functions import move_piece, place_wall_with_check, check_win, get_legal_moves, flip_board
from game.graph_search import BoardGraph
from parameters import Parameters
from game.printing import get_printable_board

parameters = Parameters()
BOARD_SIZE = parameters.board_size
START_WALLS = parameters.number_of_walls


class Player:
    def __init__(self, start_pos, number_of_walls):

        """
        Class storing information related to each player in the game

        inputs:
            start_pos: the starting position of the player
            number_of_walls: the number of walls the player should start with
        """

        self.start_pos = start_pos
        self.start_walls = number_of_walls
        self.walls = number_of_walls
        self.pos = start_pos


class Quoridor:
    def __init__(self, p1_start=np.array([0, BOARD_SIZE // 2]), p2_start=np.array([BOARD_SIZE - 1, BOARD_SIZE // 2])):

        """
        class storing an instance of the game

        inputs:
            p1_start: np.ndarray
                start position of player 1

            p2_start:
                start position of player 2
        """

        # size of board
        self.board_size = BOARD_SIZE

        # array storing state of board
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE, 4))

        # lost storing both player classes
        self.players = [Player(p1_start, START_WALLS), Player(p2_start, START_WALLS)]

        # number of players in the game
        self.num_players = len(self.players)

        # update board array to have correct player information
        self.board[p1_start[0], p1_start[1], 2] = 1
        self.board[p2_start[0], p2_start[1], 3] = 1

        # keeps track of who's move it is
        self.moving_now = 0

        # boolean tracking whether the game is currently in progress
        self.playing = True

        # number of starting walls for each player
        self.start_walls = START_WALLS

        # keeps track of the winner
        self.winner = 0

        # board and copy of the board stored in graph form (to be added later)
        self.board_graph = BoardGraph()
        self.copy_board_graph = BoardGraph()

    def move(self, command, get_time_info=False):

        """
        handles a single player move

        inputs:
            command: np.ndarray
                the move command. Takes the form [move_dir[0], move_dir[1], wall_loc[0], wall_loc[1],
                                                  wall_orientation[0], wall_orientation[1]] 

            get_time_info: bool
                boolean which returns time taken by this function if True
        """

        # set trackers for what stage of the move we are in
        moving = 0.
        illegal_move_handling = 0.
        checking_winner = 0.
        wall_handling = 0.

        # start timing
        t0 = time.time()

        # if we are not in the middle of a game stop the move now
        if not self.playing:
            return None

        # get necessary information about both players
        player_moving = self.players[self.moving_now]
        player_moving_pos = player_moving.pos

        # break up the command into relevant parts
        move_command = command[0:2]
        wall_pos_command = command[2:4]
        wall_orientation_command = command[4:6]

        # perform the move
        legal_move, jump = move_piece(self.board, player_moving_pos, move_command)
        reward = 0

        # tracker for time 
        t1 = time.time()

        # if the move is legal, carry out the movement
        if legal_move:
            if jump:
                self.players[self.moving_now].pos = 2 * move_command + player_moving_pos
            else:
                self.players[self.moving_now].pos = move_command + player_moving_pos

        # if placing a wall, handle the wall placement
        if player_moving.walls > 0:
            self.copy_board_graph.copy_graph(self.board_graph)
            legal_wall = place_wall_with_check(self.board, wall_pos_command, wall_orientation_command,
                                               self.players[0].pos, self.players[1].pos, self.copy_board_graph)

            # if the wall is legal then add it to the board
            if legal_wall:
                player_moving.walls -= 1
                self.board_graph.wall_placement(wall_pos_command, wall_orientation_command)

        # if no wall placed, record that
        else:
            legal_wall = False

        # if legal give a reward for playing a legal move
        if legal_move or legal_wall:
            legal = True
            reward += parameters.legal_move_reward

        # if no legal move was played add a penalty
        else:
            reward += parameters.illegal_move_reward
            legal = False

        # time tracker
        t2 = time.time()

        # if the move is illegal then force a random move to be played
        true_move = command
        if not legal:

            # get a list of all possible moves
            possible_moves = get_legal_moves(self.board, player_moving_pos)

            # if no moves are possible the game ends in a draw
            if len(possible_moves) == 0:
                self.playing = False
                self.winner = 0
                return self.get_state(), self.playing, self.winner, reward, False

            # if a move is legal then select randomly from legal moves
            random_move_ind = np.random.choice(len(possible_moves))
            random_move = possible_moves[random_move_ind]
            true, jump = move_piece(self.board, player_moving_pos, random_move)

            # perform the move
            if true:
                if jump:
                    self.players[self.moving_now].pos = 2 * random_move + player_moving_pos
                else:
                    self.players[self.moving_now].pos = random_move + player_moving_pos

            true_move = np.block([random_move, np.zeros(true_move.size - 2)])

        # time tracker
        t3 = time.time()

        # see if there is a winner
        winner = check_win(self.board)
        if winner != 0:
            self.playing = False
            self.winner = winner

        # change the tracker of the player who is moving
        self.moving_now = (self.moving_now + 1) % 2

        # time tracker
        t4 = time.time()

        # track time in each relevant position
        moving += t1 - t0
        wall_handling += t2 - t1
        illegal_move_handling += t3 - t2
        checking_winner += t4 - t3

        # return whatever is necessary
        if not get_time_info:
            return self.get_state(), self.playing, winner, reward, legal, true_move
        else:
            return self.get_state(), self.playing, winner, reward, legal, true_move, moving, illegal_move_handling, \
                checking_winner, wall_handling

    def get_state(self, flip=False, flatten=True):

        """
        a command to return a state vector representing the state of the board

        parameters:
            flip: bool
                True if you want to get a flipped version of the board

            flatten: bool
                True if you want the state vector to be flat
        """

        # track the number of walls currently held by each player
        p1_walls = np.zeros(self.start_walls + 1)
        p2_walls = np.zeros(self.start_walls + 1)
        p1_walls[self.players[0].walls] = 1
        p2_walls[self.players[1].walls] = 1

        # return the flat (or not flat) state vector
        if not flip:
            if flatten:
                flat_board = copy.copy(self.board).flatten()
                return np.concatenate([flat_board, p1_walls, p2_walls])
            else:
                return self.board, p1_walls, p2_walls
        else:
            if flatten:
                flat_board = flip_board(self.board, self.players[0].pos, self.players[1].pos).flatten()
                return np.concatenate([flat_board, p2_walls, p1_walls])
            else:
                return flip_board(self.board, self.players[0].pos, self.players[1].pos), p1_walls, p2_walls

    def reset(self, random_positions=False):

        """
        a function which completely resets the board

        inputs:
            random_posotions: bool
                True if you want random player positions after the reset
        """

        # if random positions randomly placeplayers and walls
        if random_positions:
            p1_start = np.array([int(np.random.choice(int(np.floor(BOARD_SIZE / 2)))),
                                 int(np.random.choice(int(BOARD_SIZE)))])
            p2_start = np.array([int(BOARD_SIZE - 1 - np.random.choice(int(np.floor(BOARD_SIZE / 2)))),
                                 int(np.random.choice(int(BOARD_SIZE)))])
            self.__init__(p1_start, p2_start)
            self.randomly_place_walls()

        # otherwise just restart like normal
        else:
            self.__init__()

    def copy_game(self, other_game):

        """
        copies this game class data into another game class

        inputs:
            other_game: Quoridor
                the game to be copied into
        """

        # get game parameters and copy them over
        self.board_size = other_game.board_size
        self.board = copy.copy(other_game.board)
        self.players = [copy.deepcopy(player) for player in other_game.players]
        self.num_players = len(self.players)
        self.moving_now = other_game.moving_now
        self.playing = other_game.playing
        self.board_graph.copy_graph(other_game.board_graph)

    def copy_board(self, board, player_1_loc, player_2_loc, player_1_walls, player_2_walls, moving_now=0):

        """
        a function to copy board and player info into this game class

        inputs:
            board: np.ndarray
                the board

            player_1_loc: np.ndarray
                position of player 1

            player_2_loc: np.ndarray
                the position of player 2

            player_1_walls: int
                the number of walls belonging to player 1

            player_2_walls: int
                the number of walls belonging to player 2

            moving_now: int
                player next to move
        """

        # do the relevant copying
        self.board = copy.copy(board)
        self.players = [Player(player_1_loc, player_1_walls), Player(player_2_loc, player_2_walls)]
        self.num_players = len(self.players)
        self.moving_now = moving_now
        self.playing = True

    def print(self):
        """
        prints a copy of the board to the command line
        """
        get_printable_board(self.board, self.players[0].walls, self.players[1].walls)

    def randomly_place_walls(self):

        """
        randomly places a bunch of legal walls on the board
        """

        # get the maximum number of walls allowed to each player
        number_of_walls_from_1 = np.random.choice(self.start_walls)
        number_of_walls_from_2 = np.random.choice(self.start_walls)

        # randomly place the necessary walls for each player
        for i in range(number_of_walls_from_1):
            self.randomly_place_single_wall(0)

        for i in range(number_of_walls_from_2):
            self.randomly_place_single_wall(1)

    def randomly_place_single_wall(self, player):

        """
        randomly places a single wall on the board from the inventory of a given player

        inputs:
            player: int
                the player whose wall we are placing
        """

        # get the location of the random wall
        loc = np.array([np.random.choice(self.board_size), np.random.choice(self.board_size)])

        # with probability half place in either orientatino
        u = np.random.uniform()
        if u > 0.5:
            orientation = np.array([0., 1.])
        else:
            orientation = np.array([1., 0.])
        self.copy_board_graph.copy_graph(self.board_graph)

        # check to see if the wall is legal and if it is, put it down
        legal = place_wall_with_check(self.board, loc, orientation, self.players[0].pos, self.players[1].pos,
                                      self.copy_board_graph)
        if legal:
            self.players[player].walls -= 1
            self.board_graph.wall_placement(loc, orientation)

    def flat_state_to_board(self, state_vector):

        """
        Converts the flattened satte-vector into board, p1_loc, p2_loc, p1_walls, p2_walls
        """

        board_part = state_vector[:4 * BOARD_SIZE**2]
        wall_part = state_vector[4 * BOARD_SIZE**2:]
        wall_part_p1 = wall_part[:START_WALLS]
        wall_part_p2 = wall_part[START_WALLS:]
        n_walls_p1 = np.where(wall_part_p1 == 1)[0]
        n_walls_p2 = np.where(wall_part_p2 == 1)[0]

        board = board_part.reshape((BOARD_SIZE, BOARD_SIZE, 4))
        p1_position = np.where(board[:, :, 2] == 1)
        p1_position = np.array([p1_position[0], p1_position[1]])
        p2_position = np.where(board[:, :, 3] == 1)
        p2_position = np.array([p2_position[0], p2_position[1]])

        return board, p1_position, p2_position, n_walls_p1, n_walls_p2
