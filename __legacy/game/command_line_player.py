import numpy as np
import copy
from game.game import Quoridor
from game.game_helper_functions import check_input_move_legal
from game.game.printing import get_printable_board
from parameters import Parameters
from game.game.move_reformatter import move_reformatter

parameters = Parameters()

class CommandLinePlayer:
    def __init__(self, agent_1='human', agent_2='human'):

        """
        Interface for running the game on the command line.

        agent_1 and agent_2 should be either 'human', representing a
        human player, or a QuoridoorAgent class.

        agent_1 moves first.
        """

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

    def human_move(self):

        """
        Handles human moves through the command line. Returns the move.
        """

        move = input('player {} move'.format(self.game.moving_now+1))
        try:
            move = move.split(",")
            move = np.array([int(i) for i in move])
            move_legal = check_input_move_legal(move)
            if not move_legal:
                print('invalid move')
                move = np.zeros(6)
        except:
            print('invalid move')
            move = np.zeros(6)

        return move

    def play(self):

        """
        Runs a single complete game.
        """

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
