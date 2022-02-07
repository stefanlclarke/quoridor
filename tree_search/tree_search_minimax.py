import numpy as np
import torch
from parameters import Parameters
from game.move_reformatter import move_reformatter
from game.game_helper_functions import check_full_move_legal
from templates.tree_search import TreeSearch
from tqdm import tqdm

parameters = Parameters()
bot_in = parameters.bot_in_dimension
bot_out = parameters.bot_out_dimension

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class MMTreeSearch(TreeSearch):
    def __init__(self, net, search_depth=3, number_checks=3, controlling=2):
        super().__init__(search_depth, number_checks, controlling)
        self.net = net

    def check_iter(self, game):
        copy_game = game

        to_move = copy_game.moving_now
        if to_move == 0:
            flip = False
        else:
            flip = True

        state = copy_game.get_state(flatten=True, flip=flip)
        values = []
        legal_moves = []

        for i in range(parameters.bot_out_dimension):
            true_move = move_reformatter(self.possible_moves[i], flip=flip)
            legal  = check_full_move_legal(copy_game.board, true_move, copy_game.players[0].pos, copy_game.players[1].pos, copy_game.players[0].walls, copy_game.players[1].walls, to_move+1)
            if legal:
                values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[i])]).to(device).float()))
                legal_moves.append(self.possible_moves[i])

        return values, legal_moves
