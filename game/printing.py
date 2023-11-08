import numpy as np


def get_printable_board(board, p1_walls, p2_walls):

    """
    prints the board in command line style

    argumensts: board: np.ndarray
        the board

    p1_walls: int
        number of walls for first player

    p2_walls: int
        number of walls for second player
    """

    # board_print is the array to be printed
    board_size = board.shape[0]
    board_print = np.array([[" " for _ in range(2 * board_size)] for _ in range(2 * board_size)], dtype='object')

    # iterate over squares and fill in what is necessary
    for i in range(board_size):
        for j in range(board_size):
            board_print[2 * j, 2 * i] = '.'
            if board[j, i][0] == 1:
                board_print[2 * j + 1, 2 * i] = "-"
            if board[j, i][1] == 1:
                board_print[2 * j, 2 * i + 1] = "|"
            if board[j, i][2] == 1:
                board_print[2 * j, 2 * i] = '1'
            if board[j, i][3] == 1:
                board_print[2 * j, 2 * i] = '2'

    # convert to a list
    board_print.tolist()

    # print in a nice way
    column_width = 1
    for row in board_print:
        row = "".join(element.ljust(column_width + 2) for element in row)
        print(row)
    print('player 1 has {} walls'.format(p1_walls))
    print('player 2 has {} walls'.format(p2_walls))
