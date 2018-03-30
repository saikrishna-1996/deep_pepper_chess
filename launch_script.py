#!/usr/bin/env python

import argparse
import copy
import multiprocessing
import time

import torch

from game_generator import generate_game
from policy_network import PolicyValNetwork_Giraffe
from self_challenge import Champion
from train import train_model

parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')

parser.add_argument('--batch-size', type=int, default=1, help='input batch size for training (default: 2500)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('--championship-rounds', type=int, default=10,
                    help='Number of rounds in the championship. Default=10')
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data', help='Path to data')
parser.add_argument('--workers', type=int, help='Number of workers used for generating games', default=4)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
torch.manual_seed(args.seed)


class GameGenerator(object):
    def __init__(self, champion: Champion, pool: multiprocessing.Pool, batch_size: int):
        self.champion = champion
        self.pool = pool
        self.batch_size = batch_size

    def play_game(self, _):
        return generate_game(self.champion.current_policy)

    def generate_games(self):
        start = time.time()
        games = self.pool.map(self.play_game, range(int(self.batch_size / args.workers + 1)))
        print("Generated {} games in {}".format(len(games), time.time() - start))
        return games

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


class PolicyImprover(object):
    def __init__(self, champion):
        self.champion = champion

    def train_model(self, new_games):
        model = copy.copy(self.champion.current_policy)
        return train_model(model=model, games=new_games, min_num_games=args.championship_rounds)

    def improve_policy(self, games):
        start = time.time()
        new_policy = self.train_model(games)
        self.champion.run_tournament(new_policy)
        print("Improved policy in: {}".format(time.time() - start))


if __name__ == '__main__':
    print("Launching Deep Pepper...")
    pool = multiprocessing.Pool(args.workers)
    champion = Champion(PolicyValNetwork_Giraffe())
    generator = GameGenerator(champion, pool, args.batch_size)
    improver = PolicyImprover(champion)

    i = 0
    while True:
        torch.save(champion.current_policy, "./{}.mdl".format(i))
        games = generator.generate_games()
        improver.improve_policy(games)
        i += 1
