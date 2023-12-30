import numpy as np
import torch
from parameters import Parameters
from game.game.move_reformatter import move_reformatter, unformatted_move_to_index
from game.game_helper_functions import check_full_move_legal
from templates.tree_search import TreeSearch
from tqdm import tqdm
from game.game.shortest_path import ShortestPathBot

parameters = Parameters()
bot_in = parameters.bot_in_dimension
bot_out = parameters.bot_out_dimension

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class MMTreeSearch(TreeSearch):
    def __init__(self, net, search_depth=3, number_checks=3, controlling=2, min_value=None, check_shortest_path=False):
        super().__init__(search_depth, number_checks, controlling)
        self.net = net
        self.min_value = min_value

        self.check_shortest_path = check_shortest_path
        if self.check_shortest_path:
            self.spbots = [ShortestPathBot(1), ShortestPathBot(2)]

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
                if self.min_value is None:
                    values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[i])]).to(device).float()))
                    legal_moves.append(self.possible_moves[i])
                else:
                    move_val = self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[i])]).to(device).float())
                    if move_val > self.min_value:
                        values.append(move_val)
                        legal_moves.append(self.possible_moves[i])

        if self.check_shortest_path:
            sp_move = self.spbots[copy_game.moving_now].move(copy_game.board)
            sp_move_ind = unformatted_move_to_index(sp_move, flip=flip)
            sp_move = np.zeros(parameters.bot_out_dimension)
            sp_move[sp_move_ind] = 1.
            values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(sp_move)]).to(device).float()))
            legal_moves.append(sp_move)

        return values, legal_moves
