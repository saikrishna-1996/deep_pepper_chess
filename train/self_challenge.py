from functools import partial
from multiprocessing import Manager

import numpy as np

from config import Config
from game.chess_env import ChessEnv
from network.policy_network import PolicyValNetwork_Giraffe
from train.MCTS import MCTS, Node

'''
Holds the best policy during training. This may be updated if the candidate shows a statistical advantage over the champion during the policy improvement phase.
'''


class Champion(object):
    def __init__(self, current_policy: PolicyValNetwork_Giraffe):
        """
        :type current_policy: PolicyValNetwork_Giraffe The initial champion.
        """
        self.current_policy = current_policy

    def test_candidate(self, candidate: PolicyValNetwork_Giraffe, pool):
        """

        :type candidate: PolicyValNetwork_Giraffe The candidate policy to test
        """
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
        root_node = Node(env, Config.EXPLORE_FACTOR)
        game_over = False

        while not game_over:
            if root_node.env.white_to_move:
                player = white
            else:
                player = black

            pi, successor, root_node = MCTS(temp=temperature, network=player, root=root_node)
            root_node = successor
            moves += 1
            game_over, z = root_node.env.is_game_over(moves, res_check=True)

        # from white perspective

        if white == candidate:
            candidate_alpha_scores.append(+z)
            incumbent_alpha_scores.append(-z)
            print("Candidate won!")
        else:
            candidate_alpha_scores.append(-z)
            incumbent_alpha_scores.append(+z)
            print("Incumbent won!")
