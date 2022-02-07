import numpy as np
import copy
from game.game import Quoridor
from game.game_helper_functions import check_input_move_legal
from game.printing import get_printable_board
from parameters import Parameters
from game.move_reformatter import move_reformatter
import pygame

parameters = Parameters()
board_size = parameters.board_size

window_size = 600
frame_rate = 30

class PygamePlayer:
    def __init__(self, agent_1='human', agent_2='human'):

        """
        Interface for running the game using pygame.

        agent_1 and agent_2 should be either 'human', representing a
        human player, or a QuoridoorAgent class.

        agent_1 moves first.
        """

        pygame.init()
        self.win = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Quoridor")
        self.font = pygame.font.SysFont('freesansbold.ttf', 80)
        self.clock = pygame.time.Clock()

        self.agent_1 = agent_1
        self.agent_2 = agent_2

        if agent_1 == 'human':
            self.agent_1_input = 'human'
            self.agent_1_output = 'true'
        else:
            self.agent_1_input = agent_1.input_type
            self.agent_1_output = agent_1.output_type

        if agent_2 == 'human':
            self.agent_2_input = 'human'
            self.agent_2_output = 'true'
        else:
            self.agent_2_input = agent_2.input_type
            self.agent_2_output = agent_2.output_type

        self.game = Quoridor()

    def draw_board(self):
        for i in range(board_size):
            for j in range(board_size):
                square_data = self.game.board[j,i,:]

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


    def human_move(self):

        """
        Handles human moves through the command line. Returns the move.
        """
        raise NotImplementedError()


    def play(self):

        """
        Runs a single complete game.
        """

        self.draw_board()

        playing = True
        while self.game.playing:
            get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)
            if self.game.moving_now == 0:
                if self.agent_1_input == 'human':
                    move = self.human_move()
                else:
                    if self.agent_1_input == 'board':
                        move = self.agent_1.move(self.game.get_state(flip=False))
                    elif self.agent_1_input == 'game':
                        move = self.agent_1.move(self.game)

                    if self.agent_1_output == 'one_hot':
                        move = move_reformatter(move, flip=False)

                    print('bot move {}'.format(move))

            if self.game.moving_now == 1:
                if self.agent_2_input == 'human':
                    move = self.human_move()
                else:
                    if self.agent_2_input == 'board':
                        move = self.agent_2.move(self.game.get_state(flip=True))
                    elif self.agent_2_input == 'game':
                        move = self.agent_2.move(self.game)

                    if self.agent_2_output == 'one_hot':
                        move = move_reformatter(move, flip=True)
                    else:
                        raise ValueError("currently agent 2 must have a one-hot output")

                    print('bot move {}'.format(move))

            new_state, playing, winner, reward, legal = self.game.move(move)

            if not self.game.playing:
                get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)
                break
