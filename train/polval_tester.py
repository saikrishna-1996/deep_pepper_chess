#Goal is to check if our current policy is any better than random policy

import argparse
import torch
import time
#from deep_pepper_chess.config import Config

parser = argparse.ArgumentParser(description='Launcher for policy tester')
parser.add_argument('--newnetwork', type=str, default=None, help='Path to the most recently trained model')
parser.add_argument('--oldnetwork', type=str, default=None, help='Path to an older trained model')
parser.add_argument('--numgames', type=int, default=100, help='how many games should they play against eachother?')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
#torch.manual_seed(args.seed)
def main():

    if args.newnetwork == None:
        new_network = torch.load(args.newnetwork)
    else:
        new_network = torch.load('./0.pt')

    #load the network and initialize with random parameters
    if args.oldnetwork == None:
        old_network = torch.load('./0.pt')
    else:
        old_network = torch.load(args.oldnetwork)


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

            pi, successor, root_node = MCTS(temp=temperature, network=player, root=root_node)
            root_node = successor
            moves = moves + 1
            game_over, z = root_node.env.is_game_over(moves)

        #from white perspective

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

    print("New network score total wins: {} Average Score: {}".format(score1,score1/args.numgames))
    print("Old network score total wins: {} Average Score: {}".format(score2,score2/args.numgames))

