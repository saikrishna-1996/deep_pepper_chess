#!/usr/bin/env python

import argparse
import multiprocessing
import os
import re
import glob

import torch

from network.policy_network import PolicyValNetwork_Giraffe
from train.game_generator import GameGenerator
from train.policy_improver import PolicyImprover
from train.self_challenge import Champion
from train.train import save_trained, load_model

parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')

parser.add_argument('--batch-size', type=int, default=2500, help='input batch size for training (default: 2500)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('--championship-rounds', type=int, default=10,
                    help='Number of rounds in the championship. Default=10')
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data', help='Path to data')
parser.add_argument('--workers', type=int, help='Number of workers used for generating games', default=2)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables GPU use')
parser.add_argument('--pretrain', action='store_true', default=True, help='Pretrain value function')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
torch.manual_seed(args.seed)


def main():
    print("Launching Deep Pepper...")
    pool = multiprocessing.Pool(args.workers)
    model, i = load_model()
    champion = Champion(model)
    generator = GameGenerator(champion, pool, args.batch_size, args.workers)
    improver = PolicyImprover(champion, args.championship_rounds)

    while True:
        games = generator.generate_games()
        improver.improve_policy(games, pool)
        i += 1
        save_trained(model,i)


if __name__ == '__main__':
    main()
