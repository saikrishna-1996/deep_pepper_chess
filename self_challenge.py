import numpy as np

from MCTS import MCTS
from chess_env import ChessEnv
# this is hypothetical functions and classes that should be created by teamates.
from config import Config


class Champion(object):
    def __init__(self, current_policy):
        self.current_policy = current_policy

    def run_tournament(self, candidate, NUMBER_GAMES=Config.NUM_GAMES):
        env = ChessEnv()

        candidate_alpha_score = []
        old_alpha_score = []

        for game_number in range(NUMBER_GAMES):
            moves = 0
            temperature = 10e-6

            p = np.random.binomial(1, 0.5) == 1
            white, black = (self.current_policy, candidate) if p else (candidate, self.current_policy)
            env.reset()
            game_over, z = env.is_game_over(moves)
            while not game_over:
                if env.white_to_move:
                    player = white
                else:
                    player = black

                pi = MCTS(env, temp=temperature, network=player)
                action_index = np.argmax(pi)
                env.step(Config.INDEXTOMOVE[action_index])
                moves += 1
                game_over, z = env.is_game_over(moves)
                # should be able to give the same state even if no room for legal move

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
