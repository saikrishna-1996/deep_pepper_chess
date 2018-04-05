from functools import partial
from multiprocessing import Manager

import numpy as np

from config import Config
from game.chess_env import ChessEnv
from train.MCTS import MCTS


class Champion(object):
    def __init__(self, current_policy):
        self.current_policy = current_policy

    def test_candidate(self, candidate, pool):
        print('START TOURNAMENT')

        with Manager() as manager:
            candidate_alpha_scores = manager.list()
            incumbent_alpha_scores = manager.list()

            games_per_worker = range(int(Config.NUM_GAMES / pool._processes + 1))
            func = partial(self.run_tournament, candidate, candidate_alpha_scores, incumbent_alpha_scores)
            pool.map(func, games_per_worker)

            candidate_total = sum(candidate_alpha_scores)
            incumbent_total = sum(incumbent_alpha_scores)

            if candidate_total > incumbent_total:
                winner = 'new_alpha'
                self.current_policy = candidate
            elif candidate_total < incumbent_total:
                winner = 'old_alpha'
            else:
                winner = None

            print("Candidate Score / Old Score / Winner: {} / {} / {}".format(candidate_total, incumbent_total, winner))

    def run_tournament(self, candidate, candidate_alpha_scores, incumbent_alpha_scores, _):
        moves = 0
        temperature = 10e-6

        p = np.random.binomial(1, 0.5) == 1
        white, black = (self.current_policy, candidate) if p else (candidate, self.current_policy)
        env = ChessEnv()
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
            candidate_alpha_scores.append(+z)
            incumbent_alpha_scores.append(-z)
            print("Candidate won!")
        else:
            candidate_alpha_scores.append(-z)
            incumbent_alpha_scores.append(+z)
            print("Incumbent won!")
