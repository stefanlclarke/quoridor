import numpy as np
import copy

class Memory:
    def __init__(self, number_other_info=0):
        """
        Class used to save information about Quoridoor games to memory.

        number_other_info: number of additional values to be saved at each iteration.
        """

        self.states = []
        self.actions = []
        self.rewards = []
        self.off_policy = []

        self.number_other_info = number_other_info
        self.other_info = [[] for _ in range(number_other_info)]

        self.game_log = []

    def reset(self):
        self.__init__(number_other_info=self.number_other_info)

    def save(self, state, action, reward, off_pol, other_info):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.off_policy.append(off_pol)

        for i in range(len(other_info)):
            self.other_info[i].append(other_info[i])

    def log_game(self):
        data = [copy.deepcopy(self.states),
                    copy.deepcopy(self.actions),
                    copy.deepcopy(self.rewards),
                    copy.copy(self.off_policy),
                    copy.copy(self.other_info)]

        self.game_log.append(data)

        self.states = []
        self.actions = []
        self.rewards = []
        self.off_policy = []
        self.other_info = [[] for _ in range(self.number_other_info)]

    def combine(self, other_memory):
        for game in other_memory.game_log:
            self.game_log.append(game)
