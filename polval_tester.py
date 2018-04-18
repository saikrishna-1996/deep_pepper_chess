# Goal is to check if our current policy is any better than random policy
import argparse
import time

import torch
from train.train import load_model
import os
import glob
from network.policy_network import PolicyValNetwork_Giraffe
import numpy as np
from game.chess_env import ChessEnv
from train.MCTS import Node, MCTS
from config import Config

parser = argparse.ArgumentParser(description='Launcher for policy tester')
parser.add_argument('--newnetwork', type=str, default=None, help='Path to the most recently trained model')
parser.add_argument('--oldnetwork', type=str, default=None, help='Path to an older trained model')
parser.add_argument('--numgames', type=int, default=100, help='how many games should they play against eachother?')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False


def main():
    old_network = PolicyValNetwork_Giraffe(pretrain=False)
    new_network, _ = load_model(args.newnetwork)
    if args.oldnetwork is None:
        list_of_files = glob.glob('./*.pt')
        if len(list_of_files) > 0:
            new_network = load_model(max(list_of_files, key=os.path.getctime))
            print('New network will be: {}'.format(new_network))
        else:
            print("No new network to test.")
            quit()

    score1 = 0
    score2 = 0
    for game in range(args.numgames):
        moves = 0
        temperature = 10e-6

        p = np.random.binomial(1, 0.5) == 1
        white, black = (new_network, old_network) if p else (new_network, old_network)

        env = ChessEnv()
        env.reset()
        root_node = Node(env, Config.EXPLORE_FACTOR)
        game_over = False

        while not game_over:
            if root_node.env.white_to_move:
                player = white
            else:
                player = black

            start = time.time()
            pi, successor, root_node = MCTS(temp=temperature, network=player, root=root_node)
            print("MCTS completed move {} in: {}".format(moves, time.time() - start))

            root_node = successor
            moves = moves + 1

            game_over, z = root_node.env.is_game_over(moves, res_check=True)

        # from white perspective

        if white == new_network:
            if z >= 1:
                score1 = score1 + 1
            else:
                score2 = score2 + 1
        else:
            if z <= -1:
                score1 = score1 + 1
            else:
                score2 = score2 + 1

    print("New network score total wins: {} Average Score: {}".format(score1, score1 / args.numgames))
    print("Old network score total wins: {} Average Score: {}".format(score2, score2 / args.numgames))


if __name__ == '__main__':
    main()
