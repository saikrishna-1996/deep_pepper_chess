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

net1 = torch.load('../0.mdl')

