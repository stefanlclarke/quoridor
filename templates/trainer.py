from game.game import Quoridor
from parameters import Parameters
import numpy as np
from models.memory import Memory
from game.move_reformatter import *
from game.shortest_path import ShortestPathBot

parameters = Parameters()
games_per_iter = parameters.games_per_iter
random_proportion = parameters.random_proportion

class Trainer:
    def __init__(self, number_other_info=2):
        """
        Template class for training AI on Quoridor.
        """

        self.game = Quoridor()
        self.memory_1 = Memory(number_other_info=number_other_info)
        self.memory_2 = Memory(number_other_info=number_other_info)
        self.spbots = [ShortestPathBot(1), ShortestPathBot(2)]

        self.possible_moves = [np.zeros(parameters.bot_out_dimension) for _ in range(parameters.bot_out_dimension)]
        for i in range(parameters.bot_out_dimension):
            self.possible_moves[i][i] = 1

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

    def play_game(self, info=None, printing=False):
        """
        Plays a game and stores all relevant information to memory.
        The agents interact with the game through the off_policy_step and
        on_policy_step functions.
        When max_rounds_per_game is reached the game is played out using
        the shortest_path policy.
        """

        rounds = 0
        unif = np.random.uniform()
        if unif < random_proportion:
            self.game.reset(random_positions=True)
        else:
            self.game.reset()
        playing = True
        while playing:
            if self.game.moving_now == 0:
                flip = False
                player = 1
                memory = self.memory_1
            else:
                flip = True
                player = 2
                rounds += 1
                memory = self.memory_2
            state = self.game.get_state(flip=flip)

            if rounds <= parameters.max_rounds_per_game:
                move, step_info, off_policy = self.on_policy_step(state, info)

            if rounds > parameters.max_rounds_per_game:
                unformatted_move = self.spbots[player-1].move(self.game.get_state(flatten=False)[0])
                move_ind = unformatted_move_to_index(unformatted_move, flip=flip)
                move = np.zeros(parameters.bot_out_dimension)
                move[move_ind] = 1
                off_policy = True
                step_info = self.off_policy_step(state, move_ind, info)

            if printing:
                print('current game state')
                self.game.print()
            new_state, playing, winner, reward, legal = self.game.move(move_reformatter(move, flip=flip))
            if printing:
                print('player {} move {} legal {}'.format(player, move_reformatter(move, flip=flip), legal))

            memory.save(state, move, reward, off_policy, step_info)

            if winner != 0:
                playing = False
                if printing:
                    game.print()
                if winner == 1:
                    self.memory_1.rewards[-1] = self.memory_1.rewards[-1] + parameters.win_reward
                    self.memory_2.rewards[-1] = self.memory_2.rewards[-1] - parameters.win_reward
                if winner == 2:
                    self.memory_1.rewards[-1] = self.memory_1.rewards[-1] - parameters.win_reward
                    self.memory_2.rewards[-1] = self.memory_2.rewards[-1] + parameters.win_reward

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

    def train(self, iterations, save_freq, name):
        """
        Runs the full training loop.

        iterations: total number of learning iterations to run
        save_freq: saves every time this many iterations have passed
        name: name to save to.
        """

        losses = 0
        for j in range(iterations//save_freq):
            for i in range(save_freq):
                self.reset_memories()
                for k in range(games_per_iter):
                    self.play_game(info=[j, i, k])
                    self.log_memories()
                loss = self.learn()
                losses += loss
            print('saving iteration {}'.format(j * save_freq))
            print('loss {}'.format(loss/save_freq))
            self.save(name, info=[j,i,k])
            losses = 0
