import numpy as np

from MCTS import MCTS
from chess_env import ChessEnv
# this is hypothetical functions and classes that should be created by teamates.
from config import Config


class Champion(object):
    def __init__(self, current_policy):
        self.current_policy = current_policy

    def self_play(self, candidate, NUMBER_GAMES=Config.NUM_GAMES):
        env = ChessEnv()
        current_board = env

        candidate_alpha_score = []
        old_alpha_score = []

        for game_number in range(NUMBER_GAMES):
            step_game = 0
            temperature = 10e-6

            p = np.random.binomial(1, 0.5) == 1
            white, black = (self.current_policy, candidate) if p else (candidate, self.current_policy)

            while not env.game_over()[0]:
                if current_board.white_to_move:
                    player = white
                else:
                    player = black

                step_game += 1
                pi = MCTS(current_board, temp=temperature, network=player)
                action_index = np.argmax(pi)
                current_board = env.step(Config.INDEXTOMOVE[action_index])
                # should be able to give the same state even if no room for legal move

            z = env.game_over()[1]
            # from white perspective

            if white == candidate:
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

        self.current_policy = winner

        # return candidate_alpha_score, old_alpha_score, winner
