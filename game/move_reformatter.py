import numpy as np
from parameters import Parameters
parameters = Parameters()


def move_reformatter(move, flip=False):
    flip_c = -(2*int(flip) - 1)
    move_part = move[0:4]
    wall_part_down = move[4:(parameters.board_size - 1)**2+4]
    wall_part_right = move[(parameters.board_size - 1)**2+4:]
    if move_part[0] == 1:
        return np.array([1,0,0,0,0,0])*flip_c
    if move_part[1] == 1:
        return np.array([-1,0,0,0,0,0])*flip_c
    if move_part[2] == 1:
        return np.array([0,1,0,0,0,0])
    if move_part[3] == 1:
        return np.array([0,-1,0,0,0,0])

    dim = parameters.board_size - 1

    if (wall_part_down == 0).all():
        move = np.where(wall_part_right==1)[0][0]
        m1 = move % dim
        m2 = (move - m1) /dim
        unflipped_move = np.array([0,0,m2,m1,1,0])
        if not flip:
            return unflipped_move
        else:
            unflipped_m1 = m1
            unflipped_m2 = m2 + 1
            m2_flipped = dim - unflipped_m2
            return np.array([0,0,m2_flipped,unflipped_m1,1,0])
    else:
        move = np.where(wall_part_down==1)[0][0]
        m1 = move % dim
        m2 = (move - m1) /dim
        unflipped_move = np.array([0,0,m2,m1,0,1])
        if not flip:
            return unflipped_move
        else:
            unflipped_m1 = m1
            unflipped_m2 = m2 + 1
            m2_flipped = dim - unflipped_m2
            return np.array([0,0,m2_flipped,unflipped_m1,0,1])

def unformatted_move_to_index(move, flip=False):
    move_part = move[0:2]
    if flip:
        flip_c = -1
    else:
        flip_c = 1
    if (move_part * flip_c == np.array([1.,0.])).all():
        return 0
    if (move_part * flip_c == np.array([-1.,0.])).all():
        return 1
    if (move_part == np.array([0.,1.])).all():
        return 2
    if (move_part == np.array([0.,-1.])).all():
        return 3

    raise ValueError("something went wrong")
