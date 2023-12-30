import numpy as np
import copy
from game.game import Quoridor
from game.game_helper_functions import check_input_move_legal
from game.game.printing import get_printable_board
from parameters import Parameters
from game.game.move_reformatter import move_reformatter
import pygame
import sys

parameters = Parameters()
board_size = parameters.board_size

window_size = 600
frame_rate = 10
square_size = window_size//board_size


class PygamePlayer:
    def __init__(self, agent_1='human', agent_2='human'):

        """
        Interface for running the game using pygame.

        agent_1 and agent_2 should be either 'human', representing a
        human player, or a QuoridoorAgent class.

        agent_1 moves first.
        """

        pygame.init()
        self.win = pygame.display.set_mode((window_size + 300, window_size))
        pygame.display.set_caption("Quoridor")
        self.font = pygame.font.SysFont('freesansbold.ttf', 30)
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

        self.first_click = None
        self.second_click = None
        self.playing = False
        self.agent_clicked = [False, False]
        self.hovering_rect_pos = None
        self.waiting_for_first_click = True

    def check_for_quit(self):
        """
        Checks to see if the player wants to quit the game, and quits.
        """

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.playing = False
                    pygame.quit()
                    sys.exit()

    def get_click_square(self):
        """
        Function for deciding if a square is being clicked on.

        Returns the coordinate of the clicked square or None.
        """

        self.clock.tick(frame_rate)
        pygame.event.pump()
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        self.check_wall_hover(mouse_pos)
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                square_coordinate = np.array(list(pos))//square_size
                wall_coordinate = self.hovering_rect_pos
                return square_coordinate, wall_coordinate
        self.check_for_quit()
        self.draw_board()
        return None, None

    def check_wall_hover(self, mouse_pos):
        """
        Function for checking if the mouse is hovering over a wall location (so
        that a wall-marker can be placed at that location).

        mouse_pos: pixel position of the mouse

        changes self.hovering_rect_pos
        """

        hover_found = False
        for i in range(board_size):
            for j in range(board_size):
                square_data = self.game.board[i,j,:]
                if (np.array([i* square_size, j*square_size - 10]) <= np.array(list(mouse_pos))).all():
                    if (np.array([(i+1)* square_size, j*square_size + 10]) >= np.array(list(mouse_pos))).all():
                        if j!= 0 and i!= board_size-1:
                            self.hovering_rect_pos = [i,j-1,0]
                            hover_found = True
                if (np.array([i* square_size - 10, j*square_size]) <= np.array(list(mouse_pos))).all():
                    if (np.array([i* square_size + 10, (j+1)*square_size]) >= np.array(list(mouse_pos))).all():
                        if i!= 0 and j!= board_size-1:
                            self.hovering_rect_pos = [i-1,j,1]
                            hover_found = True
        if not hover_found:
            self.hovering_rect_pos = None

    def draw_board(self):
        """
        Updates the pygame display and prints the board.
        """

        self.win.fill((0,0,0))
        for i in range(board_size):
            for j in range(board_size):
                square_data = self.game.board[j,i,:]

                #printing square colour
                if i % 2 == j % 2:
                    square_colour = (225,203,142)
                else:
                    square_colour = (255,255,204)
                pygame.draw.rect(self.win, square_colour, pygame.Rect((i*square_size,j*square_size), (square_size, square_size)))

                circle_centre = (i*square_size + square_size//2, j*square_size + square_size//2)
                if square_data[3] == 1:
                    if self.agent_clicked[1] == True:
                        colour = (255,100,0)
                    else:
                        colour = (255,0,0)
                    pygame.draw.circle(self.win, (100,0,0), circle_centre, square_size//3 + 5)
                    pygame.draw.circle(self.win, colour, circle_centre, square_size//3)
                if square_data[2] == 1:
                    if self.agent_clicked[0] == True:
                        colour = (0,100,255)
                    else:
                        colour = (0,0,255)
                    pygame.draw.circle(self.win, (0,0,100), circle_centre, square_size//3 + 5)
                    pygame.draw.circle(self.win, colour, circle_centre, square_size//3)

                if square_data[0] == 1:
                    pygame.draw.rect(self.win, (51, 26, 0), pygame.Rect(((i)*square_size, (j+1)*square_size - 10), (square_size, 20)))
                if square_data[1] == 1:
                    pygame.draw.rect(self.win, (51, 26, 0), pygame.Rect(((i+1)*square_size-10, (j)*square_size), (20, square_size)))

        if self.game.moving_now == 0:
            moving = 'blue'
        else:
            moving = 'red'
        text_1 = self.font.render('{}\'s move'.format(moving), True, (255,255,255))
        text_2 = self.font.render('blue has {} walls'.format(self.game.players[0].walls), True, (255,255,255))
        text_3 = self.font.render('red has {} walls'.format(self.game.players[1].walls), True, (255,255,255))
        self.win.blit(text_1, (window_size, 0))
        self.win.blit(text_2, (window_size, 30))
        self.win.blit(text_3, (window_size, 60))

        if self.hovering_rect_pos is not None and self.waiting_for_first_click:
            i = self.hovering_rect_pos[0]
            j = self.hovering_rect_pos[1]
            orientation = self.hovering_rect_pos[2]
            square_data = self.game.board[j,i,:]
            if square_data[0] != 1 and orientation == 0:
                pygame.draw.rect(self.win, (100, 100, 100), pygame.Rect(((i)*square_size, (j+1)*square_size - 10), (2*square_size, 10)))
            if square_data[1] != 1 and orientation == 1:
                pygame.draw.rect(self.win, (100, 100, 100), pygame.Rect(((i+1)*square_size-10, (j)*square_size), (10, 2*square_size)))

        pygame.display.update()

    def human_move(self):
        """
        Handles human moves through pygame. Returns the move.
        """

        waiting_for_legal_move = True
        while waiting_for_legal_move:
            self.waiting_for_first_click = True
            while self.waiting_for_first_click:
                click_coordinate, wall_coordinate = self.get_click_square()
                click_coordinate = np.flip(click_coordinate)
                wall_coordinate = np.flip(wall_coordinate)
                self.check_for_quit()
                if click_coordinate is not None and wall_coordinate is None:
                    self.first_click = click_coordinate
                    if (self.game.players[self.game.moving_now].pos == click_coordinate).all():
                        self.waiting_for_first_click = False
                        self.agent_clicked[self.game.moving_now] = True
                        self.draw_board()
                        pygame.display.update()
                elif wall_coordinate is not None:
                    pos = wall_coordinate[1:]
                    if wall_coordinate[0] == 1:
                        orientation = np.array([1.,0.])
                    else:
                        orientation = np.array([0.,1.])
                    return np.concatenate([np.array([0.,0.]), pos, orientation])


            waiting_for_second_click = True
            while waiting_for_second_click:
                click_coordinate, wall_coordinate = self.get_click_square()
                click_coordinate = np.flip(click_coordinate)
                wall_coordinate = np.flip(wall_coordinate)
                self.check_for_quit()
                if click_coordinate is not None:
                    self.second_click = click_coordinate
                    #self.clock.tick(frame_rate)
                    waiting_for_second_click = False

            move_loc = self.first_click
            move_direction = -(self.first_click - self.second_click)
            move = np.concatenate([move_direction, np.array([0., 0., 0., 0.])])
            legal = check_input_move_legal(move)
            if legal:
                waiting_for_legal_move = False
            self.agent_clicked = [False, False]
            self.draw_board()
            pygame.display.update()
        return move

    def write_calculating(self):
        text = self.font.render('!!! CALCULATING !!!', True, (255,255,255))
        self.win.blit(text, (window_size, 90))
        pygame.display.update()

    def play(self):

        """
        Runs a single complete game.
        """

        self.draw_board()
        pygame.display.update()

        self.playing = True
        while self.playing:
            get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)
            if self.game.moving_now == 0:
                if self.agent_1_input == 'human':
                    move = self.human_move()
                else:
                    self.write_calculating()
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
                    self.write_calculating()
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
            self.draw_board()
            pygame.display.update()
            self.agent_clicked = [False, False]

            if not self.game.playing:
                get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)
                break
