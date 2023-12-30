import numpy as np
from quoridor_env.config import game_config


def move_reformatter(move, board_size=game_config.BOARD_SIZE, flip=False):

    """
    converts one-hot move into cartesian move

    inputs:
        move: np.ndarray
            move in one-hot format

        flip: bool
            True if board should be flipped while evaluating

    returns: cartesian form of move
    """

    # get thing to multiply board by if flipping
    flip_c = -(2 * int(flip) - 1)

    # get move and wall parts of one-hot move
    move_part = move[0:4]
    wall_part_down = move[4:(board_size - 1)**2 + 4]
    wall_part_right = move[(board_size - 1)**2 + 4:]

    # if moving return cartesian move
    if move_part[0] == 1:
        return np.array([1, 0, 0, 0, 0, 0]) * flip_c
    if move_part[1] == 1:
        return np.array([-1, 0, 0, 0, 0, 0]) * flip_c
    if move_part[2] == 1:
        return np.array([0, 1, 0, 0, 0, 0])
    if move_part[3] == 1:
        return np.array([0, -1, 0, 0, 0, 0])

    dim = board_size - 1

    # if placing vertical wall handle that
    if (wall_part_down == 0).all():
        move = np.where(wall_part_right == 1)[0][0]
        m1 = move % dim
        m2 = (move - m1) / dim
        unflipped_move = np.array([0, 0, m2, m1, 1, 0])

        # flip correctly
        if not flip:
            return unflipped_move
        else:
            unflipped_m1 = m1
            unflipped_m2 = m2 + 1
            m2_flipped = dim - unflipped_m2
            return np.array([0, 0, m2_flipped, unflipped_m1, 1, 0])

    # if placing horizontal wall handle that
    else:
        move = np.where(wall_part_down == 1)[0][0]
        m1 = move % dim
        m2 = (move - m1) / dim
        unflipped_move = np.array([0, 0, m2, m1, 0, 1])

        # flip correctly
        if not flip:
            return unflipped_move
        else:
            unflipped_m1 = m1
            unflipped_m2 = m2 + 1
            m2_flipped = dim - unflipped_m2
            return np.array([0, 0, m2_flipped, unflipped_m1, 0, 1])


def unformatted_move_to_index(move, board_size=game_config.BOARD_SIZE, flip=False):

    """
    takes a cartesian move and returns the index for the equivalent one-hot moves

    inputs:
        move: np.ndarray
            cartesian move

    returns: index of one-hot (int)
    """

    # get relevant part
    move_part = move[0:2]

    # if flipping do it
    if flip:
        flip_c = -1
    else:
        flip_c = 1

    # check cases and return
    if (move_part * flip_c == np.array([1., 0.])).all():
        return 0
    if (move_part * flip_c == np.array([-1., 0.])).all():
        return 1
    if (move_part == np.array([0., 1.])).all():
        return 2
    if (move_part == np.array([0., -1.])).all():
        return 3

    # handle wall part
    wall_pos = move[2:4]
    wall_orientation = move[4:6]

    if flip:
        wall_pos[0] = (board_size - 1) - wall_pos[0] - 1

    binary_wall_pos = wall_pos[0] * (board_size - 1) + wall_pos[1]

    # vertical wall
    if wall_orientation[1] == 1:
        return int(4 + binary_wall_pos)

    # horizontal wall
    elif wall_orientation[0] == 1:
        return int(4 + (board_size - 1)**2 + binary_wall_pos)

    raise ValueError("something went wrong")
