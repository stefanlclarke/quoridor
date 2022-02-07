import numpy as np
import pygame
import copy

board_size = 5
window_size = 600
frame_rate = 30

square_size = window_size//board_size

pygame.init()
win = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Quoridoor")
font = pygame.font.SysFont('freesansbold.ttf', 80)
small_font = pygame.font.SysFont('freesansbold.ttf', 20)
clock = pygame.time.Clock()

#the board is [size x size x 4 big.]
example_board = np.array([[[0,1,0,0],[0,0,0,0],[0,0,1,0],[1,0,0,0],[1,0,0,0]],
                          [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]],
                          [[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]],
                          [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                          [[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]])

def draw_board(board):
    for i in range(board_size):
        for j in range(board_size):
            square_data = board[j,i,:]

            #printing square colour
            if i % 2 == j % 2:
                square_colour = (225,203,142)
            else:
                square_colour = (255,255,204)
            pygame.draw.rect(win, square_colour, pygame.Rect((i*square_size,j*square_size), (square_size, square_size)))

            circle_centre = (i*square_size + square_size//2, j*square_size + square_size//2)
            if square_data[3] == 1:
                pygame.draw.circle(win, (100,0,0), circle_centre, square_size//3 + 5)
                pygame.draw.circle(win, (255,0,0), circle_centre, square_size//3)
            if square_data[2] == 1:
                pygame.draw.circle(win, (0,0,100), circle_centre, square_size//3 + 5)
                pygame.draw.circle(win, (0,0,255), circle_centre, square_size//3)

            if square_data[0] == 1:
                pygame.draw.rect(win, (51, 26, 0), pygame.Rect(((i)*square_size, (j+1)*square_size - 10), (square_size, 20)))
            if square_data[1] == 1:
                pygame.draw.rect(win, (51, 26, 0), pygame.Rect(((i+1)*square_size-10, (j)*square_size), (20, square_size)))

def check_move_legal(board, start, direction, jump_allowed=True):
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
    if jump_allowed:
        if new_square_data[2] == 1 or new_square_data[3] == 1:
            jump_square = start + 2*direction

            if (jump_square >= board_size).any():
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
        print("ILLEGAL MOVE")
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
            possible_moves = [move for move in moves if ((point+move>=0).all() and (point+move<board_size).all())]
            for move in possible_moves:
                if visited[int(point[0] + move[0]), int(point[1] + move[1])] == 0:
                    move_legal, jump = check_move_legal(board, point, move, False)
                    if move_legal:
                        currently_checking.append(point + move)
            visited[int(point[0]), int(point[1])] = 1
            removearray(currently_checking, point)
    return visited

def check_both_players_can_reach_end(board, player_1_loc, player_2_loc):
    possible_moves_1 = get_possible_move_spaces(board, player_1_loc)
    possible_moves_2 = get_possible_move_spaces(board, player_2_loc)
    if np.linalg.norm(possible_moves_1[0]) == 0 or np.linalg.norm(possible_moves_2[-1]) == 0:
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

    new_board = copy.copy(board)
    place_wall(new_board, loc, orientation)
    end = check_both_players_can_reach_end(board, player_1_loc, player_2_loc)
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
        return ValueError('invalid command')

def place_wall_with_check(board, loc, orientation, player_1_loc, player_2_loc):
    legal = check_wall_placement_legal(board, loc, orientation, player_1_loc, player_2_loc)
    if legal:
        place_wall(board, loc, orientation)
        return True
    return False

board = example_board

player_1_loc = np.array([0,2])
player_2_loc = np.array([4,2])

#place_wall_with_check(board, np.array([1,1]), np.array([1,0]), player_1_loc, player_2_loc)
#place_wall_with_check(board, np.array([0,0]), np.array([0,1]), player_1_loc, player_2_loc)
#place_wall_with_check(board, np.array([1,0]), np.array([0,1]), player_1_loc, player_2_loc)
print(get_possible_move_spaces(board, np.array([1.,1.])))


last_press = 0

run = True
while run:
    clock.tick(frame_rate)

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    move_1 = np.zeros(2)
    move_2 = np.zeros(2)

    time = pygame.time.get_ticks()
    if time > last_press + 200:

        last_press = time
        if keys[pygame.K_w]:
            move_1[0] = -1
        if keys[pygame.K_s]:
            move_1[0] = 1
        if keys[pygame.K_a]:
            move_1[1] = -1
        if keys[pygame.K_d]:
            move_1[1] = 1

        if keys[pygame.K_u]:
            move_2[0] = -1
        if keys[pygame.K_j]:
            move_2[0] = 1
        if keys[pygame.K_h]:
            move_2[1] = -1
        if keys[pygame.K_k]:
            move_2[1] = 1

    legal, jump = move_piece(board, player_1_loc, move_1)
    if legal:
        if jump:
            player_1_loc = 2*move_1 + player_1_loc
        else:
            player_1_loc = move_1 + player_1_loc

    legal, jump = move_piece(board, player_2_loc, move_2)
    if legal:
        if jump:
            player_2_loc = 2*move_2 + player_2_loc
        else:
            player_2_loc = move_2 + player_2_loc


    draw_board(example_board)
    pygame.display.update()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
