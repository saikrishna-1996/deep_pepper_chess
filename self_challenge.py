import numpy as np

#this is hypothetical functions and classes that should be created by teamates.
import chess.uci
from policy_network import PolicyValNetwork_Full, PolicyValNetwork_Full_candidate
import value_network
from chess_env import ChessEnv
import stockfish_eval
from features import BoardToFeature
import config
from MCTS import MCTS

def game_over(state):
    if chess.is_game_over(state):

        score = chess.results(state)
        if score == 0:
            return True, -1
        if score == 0.5:
            return True, 0
        if score == 1:
            return True, 1

    else:
        return False, None


def Generating_challenge(NUMBER_GAMES, env: ChessEnv):
    current_board = env

    candidate_alpha_score=[]
    old_alpha_score=[]

    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 10e-6

        if np.random.binomial(1, 0.5) ==1:
            white = PolicyValNetwork_Full_candidate
            black = PolicyValNetwork_Full
        else:
            white = PolicyValNetwork_Full
            black = PolicyValNetwork_Full_candidate


        repition =0
        state_list = []

        while not game_over(current_board)[0]:

            state_list.append(current_board)

            if step_game % 2:

                player = black
            else:

                player = white
            step_game += 1


            pi = MCTS(current_board, init_W=[0 for i in range(64 * 63)],  # what is the shape of this pi ????????
                       init_N=[1 for i in range(64 * 63)],
            temp = temperature, explore_factor = 2,network=player, dirichlet_alpha= 0.04, epsilon=0.1)


            action_index = np.argmax(pi)

            current_board = env.step( config.INDEXTOMOVE[action_index])

            # should be able to give the same state even if no room for legal move

        z = game_over(current_board)[1]
        # from white perspective

        if white == PolicyValNetwork_Full_candidate:
            candidate_alpha_score.append(+z)
            old_alpha_score.append(-z)

        else:
            candidate_alpha_score.append(-z)
            old_alpha_score.append(+z)



    if sum(candidate_alpha_score) >sum(old_alpha_score):
        winner = 'new_alpha'

    elif sum(candidate_alpha_score) < sum(old_alpha_score):
        winner = 'old_alpha'

    else:
        winner = None
    return candidate_alpha_score,old_alpha_score,winner
