#Goal is to check if our current policy is any better than random policy

import argparse
import torch
import time
#from deep_pepper_chess.config import Config

parser = argparse.ArgumentParser(description='Launcher for policy tester')
parser.add_argument('--network1', type=str, default='0.mdl', help='choose your player')
parser.add_argument('--network2', type=str, default='whatver', help='choose your opponent')
parser.add_argument('--numgames', type=int, default=100, help='how many games should they play against eachother?')
parser.add_argument('--no-cuda', action='store_true', default=True, help='disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
#torch.manual_seed(args.seed)

net1 = torch.load('/home/saikrish/deep_pepper_chess/train/0.pt')

#load the network and initialize with random parameters
net2 = torch.load('/home/saikrish/deep_pepper_chess/train/31.pt')


score1 = 0
score2 = 0
for lol in range(args.numgames):
    moves = 0
    temperature = 10e-6

    p = np.random.binomial(1, 0.5) == 1
    white, black = (net1, net2) if p else (net2, net1)

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

    if white == net1:
        if z >= 1:
            score1 = score1 + 1
            print("first network won")
        else:
            score2 = score2 + 1
            print("second network won")
    else:
        if z <= -1:
            score1 = score1 + 1
            print("first network won")
        else:
            score2 = score2 + 1
            print("second network won")

print("final score is ")
print(score1/args.numgames)


