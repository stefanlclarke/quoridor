import numpy as np
import copy
from parameters import Parameters
from game.printing import get_printable_board

parameters = Parameters()
board_size = parameters.board_size

def check_move_legal(board, start, direction, jump_allowed=True, pass_through_player=False):
    initial_square_data = board[int(start[0]), int(start[1])]
    end = start + direction
    if (end < 0).any() or (end >= board_size).any():
        return False, False

    new_square_data = board[int(end[0]), int(end[1])]
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
    if new_square_data[2] == 1 or new_square_data[3] == 1:
        if pass_through_player:
            return True, False

        if jump_allowed:
            jump_square = start + 2*direction

            if (jump_square >= board_size).any():
                return False, False
            if (jump_square < 0).any():
                return False, False
            jump_square_data = board[int(jump_square[0]), int(jump_square[1])]

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
            return True, True
    return True, False

def get_legal_moves(board, location):
    possible_moves = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.])]
    legal_moves = []
    for move in possible_moves:
        legal, jump = check_move_legal(board, location, move, pass_through_player=True)
        if legal:
            legal_moves.append(move)
    return legal_moves

def move_piece(board, location, direction):
    if np.linalg.norm(direction) == 0:
        return False, False

    initial_square = board[int(location[0]), int(location[1])]
    if initial_square[2] == 1:
        player = 1
    elif initial_square[3] == 1:
        player = 2
    else:
        raise ValueError("no player at this location")
    move_legal, jump = check_move_legal(board, location, direction)
    if not move_legal:
        return False, False
    elif jump:
        new_square = location + 2 * direction
        initial_square[player+1] = 0
        board[int(new_square[0]), int(new_square[1])][player+1] = 1
        return True, True
    else:
        new_square = location + direction
    initial_square[player+1] = 0
    board[int(new_square[0]), int(new_square[1])][player+1] = 1
    return True, False

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def get_possible_move_spaces(board, location):
    visited = np.zeros((board_size, board_size))
    currently_checking = [location]
    moves = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.])]
    while len(currently_checking) > 0:
        for point in currently_checking:
            possible_moves = []
            for move in moves:
                if (point+move>=0).all() and (point+move<board_size).all():
                    possible_moves.append(move)
            for move in possible_moves:
                if visited[int(point[0] + move[0]), int(point[1] + move[1])] == 0:
                    move_legal, jump = check_move_legal(board, point, move, True, True)
                    if move_legal:
                        currently_checking.append(point + move)
            visited[int(point[0]), int(point[1])] = 1
            removearray(currently_checking, point)
    return visited

def check_both_players_can_reach_end(board, player_1_loc, player_2_loc):
    possible_moves_1 = get_possible_move_spaces(board, player_1_loc)
    possible_moves_2 = get_possible_move_spaces(board, player_2_loc)
    if np.linalg.norm(possible_moves_1[-1]) == 0 or np.linalg.norm(possible_moves_2[0]) == 0:
        return False
    return True

def check_wall_placement_legal(board, loc, orientation, player_1_loc, player_2_loc):
    if (loc < 0).any() or (loc >= board_size - 1).any():
        return False

    if orientation[0] == 1 and orientation[1] == 0:
        if board[int(loc[0]), int(loc[1]), 1] == 1:
            return False
        if board[int(loc[0]) + 1, int(loc[1]), 1] == 1:
            return False
        if board[int(loc[0]), int(loc[1]), 0] == 1 and board[int(loc[0]), int(loc[1])+1, 0] == 1:
            return False
    elif orientation[1] == 1 and orientation[0] == 0:
        if board[int(loc[0]), int(loc[1]), 0] == 1:
            return False
        if board[int(loc[0]), int(loc[1])+1, 0] == 1:
            return False
        if board[int(loc[0]), int(loc[1]), 1] == 1 and board[int(loc[0])+1, int(loc[1]), 1] == 1:
            return False

    new_board = copy.deepcopy(board)

    place_wall(new_board, loc, orientation)
    end = check_both_players_can_reach_end(new_board, player_1_loc, player_2_loc)
    if not end:
        return False

    return True

def place_wall(board, loc, orientation):
    if orientation[0] == 1 and orientation[1] == 0:
        board[int(loc[0]), int(loc[1]), 1] = 1
        board[int(loc[0]) + 1, int(loc[1]), 1] = 1
        return True
    elif orientation[1] == 1 and orientation[0] == 0:
        board[int(loc[0]), int(loc[1]), 0] = 1
        board[int(loc[0]), int(loc[1])+1, 0] = 1
        return True
    else:
        raise ValueError('invalid command')

def place_wall_with_check(board, loc, orientation, player_1_loc, player_2_loc):
    if np.linalg.norm(orientation) == 0:
        return False
    legal = check_wall_placement_legal(board, loc, orientation, player_1_loc, player_2_loc)
    if legal:
        place_wall(board, loc, orientation)
        return True
    return False

def check_win(board):
    layer_0 = board[0]
    layer_n = board[board_size-1]
    for piece in layer_0:
        if piece[3] == 1:
            return 2
    for piece in layer_n:
        if piece[2] == 1:
            return 1
    return 0

def check_input_move_legal(move):
    movement_part = move[0:2]
    wall_part = move[2:]
    wall_loc = wall_part[0:2]
    wall_orientation = wall_part[2:]
    if np.linalg.norm(movement_part)>0 and np.linalg.norm(wall_part)>0:
        return False
    if abs(movement_part[0])>0 and abs(movement_part[1])>0:
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
    new_board = copy.copy(board)
    new_board[int(player_1_loc[0]), int(player_1_loc[1])][2] = 0
    new_board[int(player_1_loc[0]), int(player_1_loc[1])][3] = 1
    new_board[int(player_2_loc[0]), int(player_2_loc[1])][3] = 0
    new_board[int(player_2_loc[0]), int(player_2_loc[1])][2] = 1
    new_board = np.flipud(new_board)
    for i in range(1, board_size):
        for j in range(board_size):
            if new_board[i,j,0] == 1:
                new_board[i,j,0] = 0
                new_board[i-1,j,0] = 1
    return new_board

def check_full_move_legal(board, move, p1_loc, p2_loc, p1_walls, p2_walls, player):
    syntax_legal = check_input_move_legal(move)
    if not syntax_legal:
        return False
    move_part = move[:2]
    wall_loc = move[2:4]
    wall_orientation = move[4:]
    if player == 1:
        player_loc = p1_loc
    else:
        player_loc = p2_loc
    if np.linalg.norm(move_part) > 0:
        legal, jump = check_move_legal(board, player_loc, move_part)
        if legal:
            return True
        else:
            return False

    wall_legal = check_wall_placement_legal(board, wall_loc, wall_orientation, p1_loc, p2_loc)
    if player == 1 and p1_walls > 0 and wall_legal:
        return True
    if player == 2 and p2_walls > 0 and wall_legal:
        return True

    return False
