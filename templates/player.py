import numpy as np
import time
from parameters import Parameters
from game.move_reformatter import unformatted_move_to_index, move_reformatter
parameters = Parameters()
random_proportion = parameters.random_proportion


def play_game(info, memory_1, memory_2, game, on_policy_step, off_policy_step, spbots, printing=False,
              random_start=True):

    """
    Plays a game and stores all relevant information to memory.
    The agents interact with the game through the off_policy_step and
    on_policy_step functions.
    When max_rounds_per_game is reached the game is played out using
    the shortest_path policy.
    """

    # stores amount of time spent in each phase of handling the game
    game_processing_time = 0.
    on_policy_time = 0.
    off_policy_time = 0.
    moving_time = 0.
    illegal_move_handling_time = 0.
    checking_winner_time = 0.
    wall_handling_time = 0.

    # stores number of games played
    rounds = 0
    ticks = 0

    # randomly decide whether to play the game from a random position or the start position
    if random_start:
        unif = np.random.uniform()
        if unif < random_proportion:
            game.reset(random_positions=True)
        else:
            game.reset()
    else:
        game.reset()

    # start the game
    playing = True

    # tracks legality of moves
    total_legal_moves = 0
    total_reward = 0

    while playing:
        ticks += 1

        # if printing then print
        if printing:
            game.print()

        # timer
        t0 = time.time()

        # decide who is moving
        if game.moving_now == 0:
            flip = False
            player = 1
            memory = memory_1
        else:
            flip = True
            player = 2
            rounds += 1
            memory = memory_2

        # get the game state
        state = game.get_state(flip=flip)

        # if max_rounds not yet reached allow the agent to play on policy
        if rounds <= parameters.max_rounds_per_game:
            move, step_info, off_policy = on_policy_step(state, info)
            t1 = time.time()
            on_policy_time += t1 - t0

        # if number of rounds has been reached play off-policy instead
        if rounds > parameters.max_rounds_per_game:
            unformatted_move = spbots[player - 1].move(game.get_state(flatten=False)[0],
                                                       game.board_graph)
            move_ind = unformatted_move_to_index(unformatted_move, flip=flip)
            move = np.zeros(parameters.bot_out_dimension)
            move[move_ind] = 1
            off_policy = True
            step_info = off_policy_step(state, move_ind, info)
            t1 = time.time()
            off_policy_time += t1 - t0

        # make the move
        new_state, playing, winner, reward, legal, true_move, moving, illegal_move_handling, checking_winner, \
            wall_handling \
            = game.move(move, get_time_info=True, reformat_from_onehot=True,
                        flip_reformat=flip)

        # save sate and move to memory
        true_move_index = unformatted_move_to_index(true_move, flip=flip)
        true_move = np.zeros(move.shape)
        true_move[true_move_index] = 1
        memory.save(state, move, reward, off_policy, step_info, true_move)

        # if someone has won end the game
        if winner != 0:
            playing = False
            if winner == 1:
                memory_1.rewards[-1] = memory_1.rewards[-1] + parameters.win_reward * \
                    (1 + parameters.win_speed_param / rounds)
                memory_2.rewards[-1] = memory_2.rewards[-1] - parameters.win_reward * \
                    (1 + parameters.win_speed_param / rounds)
            if winner == 2:
                memory_1.rewards[-1] = memory_1.rewards[-1] - parameters.win_reward * \
                    (1 + parameters.win_speed_param / rounds)
                memory_2.rewards[-1] = memory_2.rewards[-1] + parameters.win_reward * \
                    (1 + parameters.win_speed_param / rounds)

        # handle timing trackers
        t2 = time.time()
        game_processing_time += t2 - t1
        moving_time += moving
        illegal_move_handling_time += illegal_move_handling
        checking_winner_time += checking_winner
        wall_handling_time += wall_handling

        # handle other trackers
        total_legal_moves += int(legal)
        total_reward += reward

    if printing:
        game.print()
        print('GAME OVER')

    # return the results (timings)
    output_dict = {
        'game_processing_time': game_processing_time,
        'on_policy_time': on_policy_time,
        'off_policy_time': off_policy_time,
        'moving_time': moving_time,
        'illegal_move_handling_time': illegal_move_handling_time,
        'checking_winner_time': checking_winner_time,
        'wall_handling_time': wall_handling_time,
        'percentage_legal_moves': total_legal_moves / ticks,
        'average_reward': total_reward / ticks,
        'game_length': rounds
    }
    return output_dict