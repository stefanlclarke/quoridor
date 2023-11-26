import numpy as np
from game.game import Quoridor
from game.game_helper_functions import check_input_move_legal
from game.printing import get_printable_board
from game.move_reformatter import move_reformatter
import pygame
import sys

WINDOW_SIZE = 600
FRAME_RATE = 10
SQUARE_COLOUR_1 = (225, 203, 142)
SQUARE_COLOUR_2 = (255, 255, 204)
AGENT_1_COLOUR = (255, 0, 0)
AGENT_1_CLICK_COLOUR = (255, 100, 0)
AGENT_1_CIRCLE_COLOUR = (100, 0, 0)
AGENT_2_COLOUR = (0, 0, 255)
AGENT_2_CLICK_COLOUR = (0, 100, 255)
AGENT_2_CIRCLE_COLOUR = (0, 0, 100)
WALL_COLOUR = (51, 26, 0)
BACKGROUND_FILL = (0, 0, 0)
TEXT_COLOUR = (255, 255, 255)
HOVER_RECTANGLE_COLOUR = (100, 100, 100)


class PygamePlayer:
    def __init__(self, board_size, start_walls, agent_1='human', agent_2='human'):

        """
        Interface for running the game using pygame.

        agent_1 and agent_2 should be either 'human', representing a
        human player, or a QuoridoorAgent class.

        agent_1 moves first.
        """

        # game parameters
        self.board_size = board_size
        self.start_walls = start_walls
        self.square_size = WINDOW_SIZE // self.board_size

        # set up pygame display
        pygame.init()
        self.win = pygame.display.set_mode((WINDOW_SIZE + 300, WINDOW_SIZE))
        pygame.display.set_caption("Quoridor")
        self.font = pygame.font.SysFont('freesansbold.ttf', 30)
        self.clock = pygame.time.Clock()

        # define the two agents
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        # set agent input parameters
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

        # define the game
        self.game = Quoridor(self.board_size, self.start_walls)

        # for tracking where in the move we are at any given time
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

        # get events and quit if they are quit
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

        # advance clock and get events
        self.clock.tick(FRAME_RATE)
        pygame.event.pump()
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()
        self.check_wall_hover(mouse_pos)

        # if a click happens handle it
        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                square_coordinate = np.array(list(pos)) // self.square_size
                wall_coordinate = self.hovering_rect_pos
                return square_coordinate, wall_coordinate

        # see if we need to quit
        self.check_for_quit()

        # draw the board
        self.draw_board()

        # if nothing happening return None
        return None, None

    def check_wall_hover(self, mouse_pos):

        """
        Function for checking if the mouse is hovering over a wall location (so
        that a wall-marker can be placed at that location).

        mouse_pos: pixel position of the mouse

        changes self.hovering_rect_pos
        """

        # find the position of the square being hovered over
        hover_found = False
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (np.array([i * self.square_size, j * self.square_size - 10]) <= np.array(list(mouse_pos))).all():
                    if (np.array([(i + 1) * self.square_size, j * self.square_size + 10])
                            >= np.array(list(mouse_pos))).all():
                        if j != 0 and i != self.board_size - 1:
                            self.hovering_rect_pos = [i, j - 1, 0]
                            hover_found = True
                if (np.array([i * self.square_size - 10, j * self.square_size]) <= np.array(list(mouse_pos))).all():
                    if (np.array([i * self.square_size + 10, (j + 1) * self.square_size])
                            >= np.array(list(mouse_pos))).all():
                        if i != 0 and j != self.board_size - 1:
                            self.hovering_rect_pos = [i - 1, j, 1]
                            hover_found = True
        
        # set hover to None if none found
        if not hover_found:
            self.hovering_rect_pos = None

    def draw_board(self):
        """
        Updates the pygame display and prints the board.
        """

        # fill board in black
        self.win.fill(BACKGROUND_FILL)

        # go through squares and fill in what is necessary
        for i in range(self.board_size):
            for j in range(self.board_size):
                square_data = self.game.board[j, i, :]

                # printing square colour
                if i % 2 == j % 2:
                    square_colour = SQUARE_COLOUR_1
                else:
                    square_colour = SQUARE_COLOUR_2
                pygame.draw.rect(self.win, square_colour, pygame.Rect((i * self.square_size, j * self.square_size),
                                                                      (self.square_size, self.square_size)))

                # print agents
                circle_centre = (i * self.square_size + self.square_size // 2,
                                 j * self.square_size + self.square_size // 2)
                if square_data[3] == 1:
                    if self.agent_clicked[1]:
                        colour = AGENT_1_CLICK_COLOUR
                    else:
                        colour = AGENT_1_COLOUR
                    pygame.draw.circle(self.win, AGENT_1_CIRCLE_COLOUR, circle_centre, self.square_size // 3 + 5)
                    pygame.draw.circle(self.win, colour, circle_centre, self.square_size // 3)
                if square_data[2] == 1:
                    if self.agent_clicked[0]:
                        colour = AGENT_2_CLICK_COLOUR
                    else:
                        colour = AGENT_2_COLOUR
                    pygame.draw.circle(self.win, AGENT_2_CIRCLE_COLOUR, circle_centre, self.square_size // 3 + 5)
                    pygame.draw.circle(self.win, colour, circle_centre, self.square_size // 3)

                # print walls
                if square_data[0] == 1:
                    pygame.draw.rect(self.win, WALL_COLOUR,
                                     pygame.Rect(((i) * self.square_size, (j + 1) * self.square_size - 10),
                                                 (self.square_size, 20)))
                if square_data[1] == 1:
                    pygame.draw.rect(self.win, WALL_COLOUR,
                                     pygame.Rect(((i + 1) * self.square_size - 10, (j) * self.square_size),
                                                 (20, self.square_size)))

        if self.game.moving_now == 0:
            moving = 'blue'
        else:
            moving = 'red'

        # print text with game info
        text_1 = self.font.render('{}\'s move'.format(moving), True, TEXT_COLOUR)
        text_2 = self.font.render('blue has {} walls'.format(self.game.players[0].walls), True, TEXT_COLOUR)
        text_3 = self.font.render('red has {} walls'.format(self.game.players[1].walls), True, TEXT_COLOUR)
        self.win.blit(text_1, (WINDOW_SIZE, 0))
        self.win.blit(text_2, (WINDOW_SIZE, 30))
        self.win.blit(text_3, (WINDOW_SIZE, 60))

        # print hovered wall        
        if self.hovering_rect_pos is not None and self.waiting_for_first_click:
            i = self.hovering_rect_pos[0]
            j = self.hovering_rect_pos[1]
            orientation = self.hovering_rect_pos[2]
            square_data = self.game.board[j, i, :]
            if square_data[0] != 1 and orientation == 0:
                pygame.draw.rect(self.win, HOVER_RECTANGLE_COLOUR,
                                 pygame.Rect(((i) * self.square_size, (j + 1) * self.square_size - 10),
                                             (2 * self.square_size, 10)))
            if square_data[1] != 1 and orientation == 1:
                pygame.draw.rect(self.win, HOVER_RECTANGLE_COLOUR,
                                 pygame.Rect(((i + 1) * self.square_size - 10, (j) * self.square_size),
                                             (10, 2 * self.square_size)))

        # update the display
        pygame.display.update()

    def human_move(self):

        """
        Handles human moves through pygame. Returns the move.
        """

        # handles events while waiting for the first click
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
                        orientation = np.array([1., 0.])
                    else:
                        orientation = np.array([0., 1.])
                    return np.concatenate([np.array([0., 0.]), pos, orientation])

            # handles events while waiting for the second click
            waiting_for_second_click = True
            while waiting_for_second_click:
                click_coordinate, wall_coordinate = self.get_click_square()
                click_coordinate = np.flip(click_coordinate)
                wall_coordinate = np.flip(wall_coordinate)
                self.check_for_quit()
                if click_coordinate is not None:
                    self.second_click = click_coordinate
                    waiting_for_second_click = False

            # checks that the move is legal
            move_direction = -(self.first_click - self.second_click)
            move = np.concatenate([move_direction, np.array([0., 0., 0., 0.])])
            legal = check_input_move_legal(move, self.board_size)
            if legal:
                waiting_for_legal_move = False
            self.agent_clicked = [False, False]

            # updates the board
            self.draw_board()
            pygame.display.update()
        
        # return the move
        return move

    def write_calculating(self):

        """
        writes 'calculating' on the screen
        """

        text = self.font.render('!!! CALCULATING !!!', True, TEXT_COLOUR)
        self.win.blit(text, (WINDOW_SIZE, 90))
        pygame.display.update()

    def play(self):

        """
        Runs a single complete game.
        """

        # update display
        self.draw_board()
        pygame.display.update()

        # start playing
        self.playing = True
        while self.playing:
            get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)

            # handle move player 0
            if self.game.moving_now == 0:
                if self.agent_1_input == 'human':
                    move = self.human_move()
                    print('human move {}'.format(move))
                else:
                    self.write_calculating()
                    if self.agent_1_input == 'board':
                        move = self.agent_1.move(self.game.get_state(flip=False))
                    elif self.agent_1_input == 'game':
                        move = self.agent_1.move(self.game)

                    if self.agent_1_output == 'one_hot':
                        move = move_reformatter(move, board_size=self.board_size, flip=False)

                    print('bot move {}'.format(move))

            # handle move player 1
            if self.game.moving_now == 1:
                if self.agent_2_input == 'human':
                    move = self.human_move()
                    print('human move {}'.format(move))
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

            # excecute move
            new_state, playing, winner, reward, legal, real = self.game.move(move)
            self.draw_board()

            # update display
            pygame.display.update()
            self.agent_clicked = [False, False]

            # quit when game ends
            if not self.game.playing:
                get_printable_board(self.game.board, self.game.players[0].walls, self.game.players[1].walls)
                break
