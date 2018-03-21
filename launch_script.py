import argparse
import glob
import multiprocessing
import torch

import self_challenge
from chess_env import ChessEnv
from game_generator import generate_games
from train import train_model

parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')

parser.add_argument('--batch-size', type=int, default=2500, help='input batch size for training (default: 2500)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('--championship-rounds', type=int, default=10,
                    help='Number of rounds in the championship. Default=10')
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./celebA', help='Path to data')
parser.add_argument('--workers', type=int, help='Number of workers used for generating games', default=4)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
torch.manual_seed(args.seed)


class GameGenerator(object):
    def __init__(self, policy):
        self.policy = policy
        self.env = ChessEnv()
        self.pool = multiprocessing.Pool(args.num_workers)

    def play_game(self):
        return generate_games(self.policy, args.batch_size, self.env)

    def __call__(self):
        self.pool.map(self.play_game)


class PolicyImprover(object):
    def __init__(self, old_policy, new_games):
        self.env = ChessEnv()
        self.old_policy = old_policy
        self.new_games = new_games
        self.pool = multiprocessing.Pool(args.num_workers)

    def train_model(self):
        return train_model(model=self.old_policy, games=self.new_games, min_num_games=args.championship_rounds)

    def challenge(self, new_policy):
        return self_challenge.Generating_challenge(self.env, old_policy=self.old_policy, new_policy=new_policy)

    def __call__(self):
        new_policy = self.train_model()
        self.pool.map(self.challenge(new_policy))


def intial_policy():
    pass


if __name__ == '__main__':
    input_files = glob.glob('*.game')
    policy = intial_policy()

    generator = GameGenerator(policy)
    while True:
        games = generator(policy)
        policy = PolicyImprover(games)
