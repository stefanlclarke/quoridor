from game.game import Quoridor
from easy_test_game.easy_test_game import EasyGame
from parameters import Parameters
import numpy as np
from models.memory import Memory
from game.shortest_path import ShortestPathBot
from templates.player import play_game

parameters = Parameters()
games_per_iter = parameters.games_between_backprops
random_proportion = parameters.random_proportion
decrease_epsilon_every = parameters.decrease_epsilon_every

PLAY_QUORIDOR = parameters.play_quoridor


class Trainer:
    def __init__(self, number_other_info=2):
        """
        Template class for training AI on Quoridor.

        inputs:
            number_other_info: int
                the amount of information stored at each iteration (on top of default)
        """

        # define game
        if PLAY_QUORIDOR:
            self.game = Quoridor()
        else:
            self.game = EasyGame()

        # define memory for each bot
        self.memory_1 = Memory(number_other_info=number_other_info)
        self.memory_2 = Memory(number_other_info=number_other_info)

        # create shortest path bots (used if game times out)
        if PLAY_QUORIDOR:
            self.spbots = [ShortestPathBot(1, self.game.board_graph), ShortestPathBot(2, self.game.board_graph)]
        else:
            self.spbots = [None, None]

        # stored list of possible moves which can be made
        self.possible_moves = [np.zeros(parameters.bot_out_dimension) for _ in range(parameters.bot_out_dimension)]
        for i in range(parameters.bot_out_dimension):
            self.possible_moves[i][i] = 1

        self.decrease_epsilon_every = decrease_epsilon_every

    def handle_pre_training(self):
        """
        Function for handling anything before the playing of games begins.
        """
        pass

    def on_policy_step(self, state, info):
        """
        Funciton where the agent interacts with the game.
        """
        raise NotImplementedError()

    def off_policy_step(self, state, move_ind, info):
        """
        Updates the info when the agent policy is not being used.
        """
        raise NotImplementedError()

    def play_game(self, info=None, printing=False, random_start=True):

        """
        Plays a game and stores all relevant information to memory.
        The agents interact with the game through the off_policy_step and
        on_policy_step functions.
        When max_rounds_per_game is reached the game is played out using
        the shortest_path policy.
        """

        return play_game(info, self.memory_1, self.memory_2, self.game, self.on_policy_step, self.off_policy_step,
                         self.spbots, printing=printing, random_start=random_start)

    def reset_memories(self):
        """
        Resets memories of both players.
        """

        self.memory_1.reset()
        self.memory_2.reset()

    def log_memories(self):
        """
        Saves the most recent game played to log.
        """

        self.memory_1.log_game()
        self.memory_2.log_game()

    def learn(self):
        """
        Calculates loss based on previously played games and performs
        the gradient update step.

        Returns the loss.
        """
        raise NotImplementedError()

    def save(self, name, info=None):
        """
        Saves the network information to memory.
        """
        raise NotImplementedError()

    def train(self, iterations, save_freq, name, get_time_info=False, print_every=100):
        """
        Runs the full training loop.

        iterations: total number of learning iterations to run
        save_freq: saves every time this many iterations have passed
        name: name to save to.
        """

        print_iteration('epoch', 'move legality', 'average reward', 'game len', 'off pol %')

        # define timing trackers
        time_playing = 0.
        time_learning = 0.
        game_processing_time = 0.
        on_policy_time = 0.
        off_policy_time = 0.
        moving_time = 0.
        illegal_move_handling_time = 0.
        checking_winner_time = 0.
        wall_handling_time = 0.

        # handle pre-train (this probably does nothing)
        self.handle_pre_training()

        losses = 0

        summed_game_info = {'n': 0}

        # loop over iterations
        for j in range(iterations):
            self.reset_memories()

            # run as many games as desired
            for k in range(games_per_iter):
                game_info = self.play_game(info=[j // self.decrease_epsilon_every + 1])

                add_to_dict_sum(game_info, summed_game_info)

                # save memory
                self.log_memories()

            if j % print_every == 0:
                print_iteration(j, summed_game_info['percentage_legal_moves'] / summed_game_info['n'],
                                summed_game_info['average_reward'] / summed_game_info['n'],
                                summed_game_info['game_length'] / summed_game_info['n'],
                                summed_game_info['percentage_moves_off_policy'] / summed_game_info['n'])
                summed_game_info = {'n': 0}

            # do backpropagation
            loss = self.learn()
            losses += loss

            # save the model
            if j % save_freq == 0:
                print('saving iteration {}'.format(j * save_freq))
                print('loss {}'.format(loss / save_freq))
                print_iteration('epoch', 'move legality', 'average reward', 'game len', 'off pol %')
                self.save(name, info=[j])
            losses = 0

        # if time info is requested return it
        if get_time_info:
            return time_playing, time_learning, game_processing_time, on_policy_time, off_policy_time, moving_time, \
                illegal_move_handling_time, checking_winner_time, wall_handling_time

    def save_most_recent_play(self, name):
        p1_actions = self.memory_1.game_log[-1][5]
        p2_actions = self.memory_2.game_log[-1][5]
        p1_actions = np.block([[g] for g in p1_actions])
        p2_actions = np.block([[g] for g in p2_actions])
        p1_rewards = self.memory_1.game_log[-1][2]
        p2_rewards = self.memory_2.game_log[-1][2]
        p1_rewards = np.block([[g] for g in p1_rewards])
        p2_rewards = np.block([[g] for g in p2_rewards])
        np.savetxt("game_samples/{}moves_p1.csv".format(name), p1_actions, delimiter=",")
        np.savetxt("game_samples/{}moves_p2.csv".format(name), p2_actions, delimiter=",")
        np.savetxt("game_samples/{}rewards_p1.csv".format(name), p1_rewards, delimiter=",")
        np.savetxt("game_samples/{}rewards_p2.csv".format(name), p2_rewards, delimiter=",")


def print_iteration(*args):
    printstring = ''
    for arg in args:
        printstring += str(arg).ljust(10)[0:10] + '\t\t'
    print(printstring)


def add_to_dict_sum(game_info, summed_game_info):
    for key in game_info.keys():
        if key in list(summed_game_info.keys()):
            summed_game_info[key] += game_info[key]
        else:
            summed_game_info[key] = game_info[key]
    summed_game_info['n'] += 1