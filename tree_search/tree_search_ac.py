import numpy as np
import torch
from game.game import Quoridor
from parameters import Parameters
from models.q_models import QNetBot
from game.move_reformatter import move_reformatter
from game.printing import get_printable_board
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

class ACTreeSearch(TreeSearch):
    def __init__(self, net, actor, search_depth=3, number_checks=3, controlling=2):
        super().__init__(search_depth, number_checks, controlling)
        self.net = net
        self.actor = actor

    def check_iter(self, game):
        copy_game = game

        to_move = copy_game.moving_now
        if to_move == 0:
            flip = False
        else:
            flip = True

        state = copy_game.get_state(flatten=True, flip=flip)

        actor_move, actor_probabilities, actor_prob = self.actor.move(torch.from_numpy(state).to(device).float())
        np_probs = actor_probabilities.cpu().detach().numpy()
        highest_actor_probs = np.argpartition(np_probs, -self.number_checks)[-self.number_checks:]
        probabilities_sorted = sorted(np_probs)
        arguments_sorted = np.argsort(np_probs)

        values = []
        legal_moves = []
        checked_legal_moves = 0

        for arg in reversed(arguments_sorted):
            true_move = move_reformatter(self.possible_moves[arg], flip=flip)
            legal  = check_full_move_legal(copy_game.board, true_move, copy_game.players[0].pos, copy_game.players[1].pos, copy_game.players[0].walls, copy_game.players[1].walls, to_move+1)
            if legal:
                values.append(self.net.forward(torch.cat([torch.from_numpy(state), torch.from_numpy(self.possible_moves[arg])]).to(device).float()))
                legal_moves.append(self.possible_moves[arg])
                checked_legal_moves += 1
            if checked_legal_moves >= self.number_checks:
                break

        return values, legal_moves
