import multiprocessing as mp
import time

import numpy as np

from config import Config
from game.chess_env import ChessEnv
from game.features import board_to_feature
from network.policy_network import PolicyValNetwork_Giraffe
from train.MCTS import MCTS, Node
from train.self_challenge import Champion

'''
The Game Generator is responsible for generating a history of games (consisting of a list of moves) through self-play. This game generator is completely parallelizable. Callers are able to set workers to the number of available CPU cores, and it will use each core to play an independent game. These will be then used to train a new champion, which will generate new games, and so forth.
'''

class GameGenerator(object):
    def __init__(self, champion: Champion, pool: mp.Pool, batch_size: int, workers: int):
        """
        Takes the most recent champion (which should correspond to the best policy), a multiprocessing pool on which to run games, and generates a number of games equal to batch_size.

        :type champion: Champion The "best" policy up to a certain point
        :type pool: mp.Pool This is the multiprocessing pool which games are generated from
        :type batch_size: int The number of games to generate
        :type workers: int The number of workers to run in parallel
        """

        self.champion = champion
        self.pool = pool
        self.workers = workers
        self.batch_size = batch_size

    def generate_game(self, model: PolicyValNetwork_Giraffe):
        np.random.seed()
        triplets = []
        step_game = 0
        temperature = 1
        # env = ChessEnv()
        # env.reset()
        game_over = False
        moves = 0
        # game_over, z = env.is_game_over(moves)
        env = ChessEnv()
        env.reset()
        root_node = Node(env, Config.EXPLORE_FACTOR)
        while not game_over:
            moves += 1
            step_game += 1
            if step_game == 50:
                temperature = 10e-6

            start = time.time()
            pi, successor, root_node = MCTS(temp=temperature, network=model, root=root_node)
            #print("Calculated next move in {}ms".format(time.time() - start))
            feature = board_to_feature(root_node.env.board)
            triplets.append([feature, pi])
            #print('')
            #print(root_node.env.board)
            #print("Running on {} ".format(mp.current_process()))
            root_node = successor
            game_over, z = root_node.env.is_game_over(moves, res_check=True)
        for i in range(len(triplets) - step_game, len(triplets)):
            triplets[i].append(z)

        return triplets

    def play_game(self, _):
        return self.generate_game(self.champion.current_policy)

    def generate_games(self):
        start = time.time()
        games = self.pool.map(self.play_game, range(self.batch_size))
        print("Generated {} games in {}".format(len(games), time.time() - start))
        return games

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
