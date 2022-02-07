import numpy as np
import torch
from game.game import Quoridor
from parameters import Parameters
from game.move_reformatter import move_reformatter
from game.printing import get_printable_board
from game.game_helper_functions import check_full_move_legal
from tqdm import tqdm
from templates.agent import QuoridoorAgent

parameters = Parameters()
bot_in = parameters.bot_in_dimension
bot_out = parameters.bot_out_dimension

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

possible_moves = [np.zeros(parameters.bot_out_dimension) for _ in range(parameters.bot_out_dimension)]
values = []
for i in range(parameters.bot_out_dimension):
    possible_moves[i][i] = 1

def split_move_array(array):
    length = len(array)
    number_moves = int(len(array)/bot_out)
    move_list = []
    for i in range(number_moves):
        move = array[i*bot_out:(i+1)*bot_out]
        move_list.append(move)
    return move_list

class TreeSearch(QuoridoorAgent):
    def __init__(self, search_depth=3, number_checks=3, controlling=2):
        super().__init__(input_type='game')
        self.search_dict = {}
        self.child_dict = {}
        self.parent_dict = {}
        self.state_dict = {}
        self.search_depth = search_depth
        self.number_checks = number_checks
        self.controlled_player = controlling - 1
        self.possible_moves = possible_moves

    def check_iter(self, game):
        raise NotImplementedError()

    def search_iter(self, game, search_dict, child_dict, parent_dict, sequence_to_expand, number_to_check, controlling):
        copy_game = Quoridor()
        copy_game.copy_game(game)
        move_list = split_move_array(sequence_to_expand)
        if len(move_list) > 1:
            prev_seq = np.concatenate(move_list[:-1]).tobytes()
            last_state_info = self.state_dict[prev_seq]
            copy_game.copy_board(*last_state_info)
        #for i in range(len(move_list)):

        if len(move_list) > 0:
            move_to_make = move_list[-1]
            to_move = copy_game.moving_now
            if to_move == 0:
                flip = False
            else:
                flip = True
            copy_game.move(move_reformatter(move_to_make, flip=flip))


        to_move = copy_game.moving_now
        if to_move == 0:
            flip = False
        else:
            flip = True
        state = copy_game.get_state(flatten=True, flip=flip)

        values, legal_moves = self.check_iter(copy_game)

        if len(values) == 0:
            search_dict[sequence_to_expand.tobytes()] = -1.
            return []

        np_vals = torch.cat(values).detach().cpu().numpy()
        number_to_check_ = min(number_to_check, len(legal_moves))
        best_move_args = np.argpartition(np_vals, -number_to_check_)[-number_to_check_:]
        best_arg = np.argmax(np_vals)
        search_dict[sequence_to_expand.tobytes()] = np_vals[best_arg]

        if copy_game.winner == 1 and controlling == 0:
            search_dict[sequence_to_expand.tobytes()] = 100.
        elif copy_game.winner == 2 and controlling == 1:
            search_dict[sequence_to_expand.tobytes()] = 100.
        elif copy_game.winner == 1 and controlling == 1:
            search_dict[sequence_to_expand.tobytes()] = -100.
        elif copy_game.winner == 2 and controlling == 0:
            search_dict[sequence_to_expand.tobytes()] = -100.

        considered_moves = [legal_moves[arg] for arg in best_move_args]
        next_keys = [np.concatenate([sequence_to_expand, move]) for move in considered_moves]

        keys_to_check = []
        for i in range(len(next_keys)):
            search_dict[next_keys[i].tobytes()] = None
            keys_to_check.append(next_keys[i].tobytes())
        child_dict[sequence_to_expand.tobytes()] = keys_to_check
        for key in keys_to_check:
            parent_dict[key] = sequence_to_expand.tobytes()

        self.state_dict[sequence_to_expand.tobytes()] = [copy_game.board, copy_game.players[0].pos, copy_game.players[1].pos, copy_game.players[0].walls, copy_game.players[1].walls]
        return keys_to_check

    def move(self, game):
        self.search_dict = {}
        keys_to_check = []
        keys_to_check += self.search_iter(game, self.search_dict, self.child_dict, self.parent_dict, np.array([]), self.number_checks, self.controlled_player)
        initial_moves = [np.frombuffer(key, dtype=np.float64) for key in keys_to_check]

        print('forward search')
        for i in tqdm(range(self.search_depth * 2)):
            new_keys = []
            for key in keys_to_check:
                move_string = np.frombuffer(key, dtype=np.float64)
                new_keys += self.search_iter(game, self.search_dict, self.child_dict, self.parent_dict, move_string, self.number_checks, self.controlled_player)
            prev_keys = keys_to_check
            keys_to_check = new_keys

        print('backward search')
        for i in tqdm(range(self.search_depth * 2)):
            next_check = []
            for key in prev_keys:
                parent = self.parent_dict[key]
                if parent not in next_check:
                    next_check.append(parent)

            for parent in next_check:
                children = self.child_dict[parent]
                child_values = [self.search_dict[child] for child in children]

                if i % 2 == 0:
                    parent_value = min(child_values)
                else:
                    parent_value = max(child_values)

                self.search_dict[parent] = parent_value

            prev_keys = next_check

        initial_move_new_values = [self.search_dict[move.tobytes()] for move in initial_moves]
        argmax = np.argmax(initial_move_new_values)

        return initial_moves[argmax]

    def print(self):
        for k,v in self.search_dict.items():
            print(np.frombuffer(k, dtype=np.float64), v)
