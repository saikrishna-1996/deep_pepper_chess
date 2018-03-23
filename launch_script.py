#!/usr/bin/env python

import argparse
import copy
import multiprocessing

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
parser.add_argument('--workers', type=int, help='Number of workers used for generating games', default=2)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
torch.manual_seed(args.seed)


class GameGenerator(object):
    def __init__(self, policy: PolicyValNetwork_Giraffe, pool: multiprocessing.Pool, batch_size: int):
        self.policy = policy
        self.pool = pool
        self.batch_size = batch_size

    def play_game(self):
        return generate_game(self.policy)

    def __call__(self, policy):
        self.policy = policy
        ongoing_games = [self.pool.apply_async(self.play_game) for i in range(self.batch_size)]
        return [g.get() for g in ongoing_games]


class PolicyImprover(object):
    def __init__(self, old_policy):
        self.old_policy = old_policy
        self.champion = Champion(old_policy)

    def train_model(self, new_games):
        return train_model(model=self.old_policy, games=new_games, min_num_games=args.championship_rounds)

    def run_challenge(self, new_policy):
        return self.champion.self_play(new_policy)

    def __call__(self, games):
        new_policy = self.train_model(games)
        self.champion.self_play(new_policy)
        return self.champion.current_policy


if __name__ == '__main__':
    print("hello")
    pool = multiprocessing.Pool(args.workers)
    policy = PolicyValNetwork_Giraffe()
    generator = GameGenerator(policy, pool, args.batch_size)
    improver = PolicyImprover(policy)

    i = 0
    while True:
        if i % 100:
            torch.save(policy, "./{}.mdl".format(i))
        games = generator(copy.copy(policy))
        policy = improver(games)
        i += 1
