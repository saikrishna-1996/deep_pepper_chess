import numpy as np

from MCTS import MCTS
from chess_env import ChessEnv
# this is hypothetical functions and classes that should be created by teamates.
from config import Config
from policy_network import PolicyValNetwork_Giraffe


def self_play(env: ChessEnv, old_policy, new_policy, NUMBER_GAMES=100):
    current_board = env

    candidate_alpha_score = []
    old_alpha_score = []

    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 10e-6

        if np.random.binomial(1, 0.5) == 1:
            white = PolicyValNetwork_Giraffe_candidate
            black = PolicyValNetwork_Giraffe
        else:
            white = PolicyValNetwork_Giraffe
            black = PolicyValNetwork_Giraffe_candidate

        repition = 0
        state_list = []

        while not env.game_over()[0]:

            state_list.append(current_board)

            if step_game % 2:
                player = black
            else:
                player = white

            step_game += 1
            pi = MCTS(current_board, temp=temperature, network=player)
            action_index = np.argmax(pi)
            current_board = env.step(Config.INDEXTOMOVE[action_index])

            # should be able to give the same state even if no room for legal move

        z = env.game_over()[1]
        # from white perspective

        if white == PolicyValNetwork_Full_candidate:
            candidate_alpha_score.append(+z)
            old_alpha_score.append(-z)

        else:
            candidate_alpha_score.append(-z)
            old_alpha_score.append(+z)

    if sum(candidate_alpha_score) > sum(old_alpha_score):
        winner = 'new_alpha'
    elif sum(candidate_alpha_score) < sum(old_alpha_score):
        winner = 'old_alpha'
    else:
        winner = None

    return candidate_alpha_score, old_alpha_score, winner
