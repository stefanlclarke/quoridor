import numpy as np
import copy


def check_move_legal(board, start, direction, jump_allowed=True, pass_through_player=False):

    """
    function checking to see if a move is legal

    inputs:
        board: np.ndarray
            the game board

        start: np.ndarray
            start position of the move (player position)

        direction: np.ndarray
            the move direction

        jump_allowed: bool
            true if we are allowing the player to hop the enemy

        pass_through_player: bool
            true if we are allowing the player to pass through the enemy

    returns:
        move_legality: bool
            True if move legal

        jump_status: bool
            True if a jump is made
    """

    # get board dimension
    board_size = board.shape[0]

    # get start and end positions for the move
    initial_square_data = board[int(start[0]), int(start[1])]
    end = start + direction

    # if moving outside of the board return False, False
    if (end < 0).any() or (end >= board_size).any():
        return False, False

    # get data for the destination square
    new_square_data = board[int(end[0]), int(end[1])]

    # check to see if a wall is blocking the movement
    if direction[1] == 1:
        if initial_square_data[1] == 1:
            return False, False
    if direction[0] == 1:
        if initial_square_data[0] == 1:
            return False, False
    if direction[1] == -1:
        if new_square_data[1] == 1:
            return False, False
    if direction[0] == -1:
        if new_square_data[0] == 1:
            return False, False

    # check to see if a player is blocking the movement
    if new_square_data[2] == 1 or new_square_data[3] == 1:
        if pass_through_player:
            return True, False

        # if we can jump, update the move destination to post-jump
        if jump_allowed:
            jump_square = start + 2 * direction

            # if jumping outside the board return False, False
            if (jump_square >= board_size).any():
                return False, False
            if (jump_square < 0).any():
                return False, False
            jump_square_data = board[int(jump_square[0]), int(jump_square[1])]

            # if the move is blocked by a wall return False, False
            if direction[1] == 1:
                if new_square_data[1] == 1:
                    return False, False
            if direction[0] == 1:
                if new_square_data[0] == 1:
                    return False, False
            if direction[1] == -1:
                if jump_square_data[1] == 1:
                    return False, False
            if direction[0] == -1:
                if jump_square_data[0] == 1:
                    return False, False

            # move is legal and jumping
            return True, True
    # move is legal and not jumping
    return True, False


def get_legal_moves(board, location):

    """
    function returning a list of possible legal moves for player

    inputs:
        board: np.ndarray
            the game board

        location: np.ndarray
            the location of the player moving

    returns:
        legal_moves: list
            list of legal moves
    """

    # list of all possible moves
    possible_moves = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]), np.array([0., -1.])]

    # list to store legal moves
    legal_moves = []

    # iterate over moves, check to see if they are legal
    for move in possible_moves:
        legal, jump = check_move_legal(board, location, move, pass_through_player=True)
        if legal:
            legal_moves.append(move)

    # return the list of legal moves
    return legal_moves


def move_piece(board, location, direction):

    """
    handles the movement of a piece on the board

    inputs:
        board: np.ndarray
            the board

        location: np.ndarray
            the start location for the move

        direction: np.ndarray
            the movement direction

    returns:
        move_legal: bool
            true if legal move

        jump: bool
            true if jump made
    """

    # if the move is a bunch of zeroes it is not legal
    if np.linalg.norm(direction) == 0:
        return False, False

    # work out which player is being moved
    initial_square = board[int(location[0]), int(location[1])]
    if initial_square[2] == 1:
        player = 1
    elif initial_square[3] == 1:
        player = 2
    else:
        raise ValueError("no player at this location")

    # check to see if the move is legal
    move_legal, jump = check_move_legal(board, location, direction)

    # if move is not legal do nothing
    if not move_legal:
        return False, False

    # if jumping update the board with the jumped move
    elif jump:
        new_square = location + 2 * direction
        initial_square[player + 1] = 0
        board[int(new_square[0]), int(new_square[1])][player + 1] = 1
        return True, True

    # if not jumping update the board with the non-jumped move
    else:
        new_square = location + direction

    # update board and return
    initial_square[player + 1] = 0
    board[int(new_square[0]), int(new_square[1])][player + 1] = 1
    return True, False


def removearray(L, arr):

    """
    Function designed to remove an array arr from a list L

    inputs:
        L: list
            the list to be searched over

        arr:
            the array which is potentially contained within list which should be removed
    """

    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def get_possible_move_spaces(board, location):

    """
    finds all positions in the board accessible from a given start position

    inputs:
        board: np.ndarray
            the board

        location: np.ndarray
            the starting location

    returns:
        visited: np.ndarray
            board-shaped array with a 1 in all positions which can be reached
    """

    # get board dimension
    board_size = board.shape[0]

    # create an array to keep track of locations which can be visited
    visited = np.zeros((board_size, board_size))

    # keeps track of places yet to be checked
    currently_checking = [location]
    moves = [np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]), np.array([0., -1.])]

    # start checking over all possible moves
    while len(currently_checking) > 0:
        for point in currently_checking:
            possible_moves = []

            # get possible move from current check location
            for move in moves:
                if (point + move >= 0).all() and (point + move < board_size).all():
                    possible_moves.append(move)

            # check to see which of these moves are legal
            for move in possible_moves:
                if visited[int(point[0] + move[0]), int(point[1] + move[1])] == 0:
                    move_legal, jump = check_move_legal(board, point, move, True, True)

                    # if it's legal check this move next
                    if move_legal:
                        currently_checking.append(point + move)

            # if delete the current position from places to check
            visited[int(point[0]), int(point[1])] = 1
            removearray(currently_checking, point)

    # return the array of places we can visit
    return visited


def check_both_players_can_reach_end(board, player_1_loc, player_2_loc):

    """
    function to make sure that both players can reach their target destination

    inputs:
        board: np.ndarray
            the board

        player_1_loc: np.ndarray
            the location of player 1

        player_2_loc: np.ndarray
            the location of player 2

    returns: True if possible, False if not
    """

    # get arrays of possible moves
    possible_moves_1 = get_possible_move_spaces(board, player_1_loc)
    possible_moves_2 = get_possible_move_spaces(board, player_2_loc)

    # check that both players can reach the end
    if np.linalg.norm(possible_moves_1[-1]) == 0 or np.linalg.norm(possible_moves_2[0]) == 0:
        return False
    return True


def check_wall_placement_legal(board, loc, orientation, player_1_loc, player_2_loc, board_graph_copy):

    """
    function to see if the placement of a wall is legal

    inputs:
        board: np.ndarray
            the board

        loc: np.ndarray
            the locatino of the wall

        orientation: np.ndarray
            the wall orientation

        player_1_loc: np.ndarray
            the position of player 1 on the board

        player_2_loc: np.ndarray
            the position of player 2 onb the boar

        board_graph_copy: Quoridor
            the spare (copy) board which we will place the wall on to see  if it is legal

        returns: True if legal, False if not
    """

    # get board dimension
    board_size = board.shape[0]

    # if placed outside board it is illegal
    if (loc < 0).any() or (loc >= board_size - 1).any():
        return False

    # check to see if any of the positions occupied by the wall are taken and return false if true
    if orientation[0] == 1 and orientation[1] == 0:
        if board[int(loc[0]), int(loc[1]), 1] == 1:
            return False
        if board[int(loc[0]) + 1, int(loc[1]), 1] == 1:
            return False
        if board[int(loc[0]), int(loc[1]), 0] == 1 and board[int(loc[0]), int(loc[1]) + 1, 0] == 1:
            return False
    elif orientation[1] == 1 and orientation[0] == 0:
        if board[int(loc[0]), int(loc[1]), 0] == 1:
            return False
        if board[int(loc[0]), int(loc[1]) + 1, 0] == 1:
            return False
        if board[int(loc[0]), int(loc[1]), 1] == 1 and board[int(loc[0]) + 1, int(loc[1]), 1] == 1:
            return False
    
    # add the wall to the copy board and check both players can reach the end
    board_graph_copy.wall_placement(loc, orientation)
    end = board_graph_copy.check_both_players_can_reach_end(player_1_loc, player_2_loc)

    # if they can't then return False, otherwise True
    if not end:
        return False
    return True


def place_wall(board, loc, orientation):

    """
    handles the actual placing of a wall on the board

    board: np.ndarray
        the board

    loc: np.ndarray
        the location of the wall

    orientation: np.ndarray
        the orientation
    """

    # put the wall on the board
    if orientation[0] == 1 and orientation[1] == 0:
        board[int(loc[0]), int(loc[1]), 1] = 1
        board[int(loc[0]) + 1, int(loc[1]), 1] = 1
        return True
    elif orientation[1] == 1 and orientation[0] == 0:
        board[int(loc[0]), int(loc[1]), 0] = 1
        board[int(loc[0]), int(loc[1]) + 1, 0] = 1
        return True
    else:
        raise ValueError('invalid command')


def place_wall_with_check(board, loc, orientation, player_1_loc, player_2_loc, board_graph_copy):

    """
    a function which checks that a wall placement is legal and then puts it on the board if so

    inputs:
        board: np.ndarray
            the board

        loc: np.ndarray
            the location of the wall

        orientation: np.ndarray
            wall orientation

        player_1_loc: np.ndarray
            the position of player 1

        player_2_loc: np.ndarray
            the position of player 2

        board_graph_copy: Quoridor
            the copy of the board which can be checked on

    returns: False if illegal, True otherwise
    """

    # if nothing in the move it is illegal
    if np.linalg.norm(orientation) == 0:
        return False

    # check legality of placement
    legal = check_wall_placement_legal(board, loc, orientation, player_1_loc, player_2_loc, board_graph_copy)

    # if legal put it down
    if legal:
        place_wall(board, loc, orientation)
        return True
    return False


def check_win(board):

    """
    a function to see if either player has won the game

    inputs:
        board: np.ndarray
            the board

    returns: 1 if p1 wins, 2 if p2, 0 otherwise
    """

    # get board dimension
    board_size = board.shape[0]

    # get important layer (the one with player info)
    layer_0 = board[0]
    layer_n = board[board_size - 1]

    # return the winner
    for piece in layer_0:
        if piece[3] == 1:
            return 2
    for piece in layer_n:
        if piece[2] == 1:
            return 1
    return 0


def check_input_move_legal(move, board_size):

    """
    checks to see if a move has the correct syntax

    inputs:
        move: np.ndarray
            the move

    returns: True if synrtax good, False otherwise
    """

    # decompose move into relevant parts
    movement_part = move[0:2]
    wall_part = move[2:]
    wall_loc = wall_part[0:2]
    wall_orientation = wall_part[2:]

    # check syntax
    if np.linalg.norm(movement_part) > 0 and np.linalg.norm(wall_part) > 0:
        return False
    if abs(movement_part[0]) > 0 and abs(movement_part[1]) > 0:
        return False
    if np.linalg.norm(movement_part) > 1:
        return False
    if (wall_loc < 0).any() or (wall_loc > board_size - 1).any():
        return False
    if np.linalg.norm(wall_orientation) > 1:
        return False
    if np.linalg.norm(move) == 0:
        return False
    return True


def flip_board(board, player_1_loc, player_2_loc):

    """
    function which returns a copy of the board which has been flipped (so the players change roles)

    inputs:
        board: np.ndarray
            the board

        player_1_loc: np.ndarray
            the location of player 1

        player_2_loc: np.ndarray
            the location of player 2

    returns:
        the new copied flipped board
    """

    # get board dimension
    board_size = board.shape[0]

    # make a copy
    new_board = copy.copy(board)

    # flip player roles
    new_board[int(player_1_loc[0]), int(player_1_loc[1])][2] = 0
    new_board[int(player_1_loc[0]), int(player_1_loc[1])][3] = 1
    new_board[int(player_2_loc[0]), int(player_2_loc[1])][3] = 0
    new_board[int(player_2_loc[0]), int(player_2_loc[1])][2] = 1
    new_board = np.flipud(new_board)

    # flip player positions
    for i in range(1, board_size):
        for j in range(board_size):
            if new_board[i, j, 0] == 1:
                new_board[i, j, 0] = 0
                new_board[i - 1, j, 0] = 1

    # return the board
    return new_board


def check_full_move_legal(board, move, p1_loc, p2_loc, p1_walls, p2_walls, player):

    """
    checks the syntax, movement part, and wall placement part of a move

    inputs:
        board: np.ndarray
            the board

        move: np.ndarray
            the move to be carried out

        p1_loc: np.ndarray
            the location of player 1

        p2_loc: np.ndarray
            the location of player 2

        p1_walls: int
            number of walls for player 1

        p2_walls: int
            number of walls for player 2

        player: int
            the player moving
    """

    # get board dimension
    board_size = board.shape[0]

    # check syntax
    syntax_legal = check_input_move_legal(move, board_size)
    if not syntax_legal:
        return False

    # decompose
    move_part = move[:2]
    wall_loc = move[2:4]
    wall_orientation = move[4:]

    # get moving player location
    if player == 1:
        player_loc = p1_loc
    else:
        player_loc = p2_loc

    # if it is a movement check the movement
    if np.linalg.norm(move_part) > 0:
        legal, jump = check_move_legal(board, player_loc, move_part)
        if legal:
            return True
        else:
            return False

    # if it is a wall placement check the wall placement
    wall_legal = check_wall_placement_legal(board, wall_loc, wall_orientation, p1_loc, p2_loc)
    if player == 1 and p1_walls > 0 and wall_legal:
        return True
    if player == 2 and p2_walls > 0 and wall_legal:
        return True

    # if you are here something is wrong
    return False
