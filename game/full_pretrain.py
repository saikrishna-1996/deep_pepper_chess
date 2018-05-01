import os
import sys

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from random import shuffle
from network.policy_network import PolicyValNetwork_Giraffe
import chess.pgn
import chess.uci
# import random
import torch
import pickle
from game.chess_env import ChessEnv
from torch.autograd import Variable
from config import Config
from game.features import board_to_feature
from game.stockfish import Stockfish
from logger import Logger
import parallel_mcts_test
import argparse
# set the logger
logger = Logger('./logs')
model = PolicyValNetwork_Giraffe(pretrain=False)



parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')
parser.add_argument('--load-path', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='path of files')

args = parser.parse_args()
def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets.double() * torch.log(pred).double(), 1))



def save_trained(model, iteration):
    torch.save(model.state_dict(), "./{}.pt".format(iteration))


def pretrain(model,boards):


    iters = 0
    feature_batch = []
    targets_val_batch = []
    targets_pol_batch = []
    shuffle(boards)
    print("Pretraining on {} board positions...".format(len(boards)))
    stockfish = Stockfish()

    for batch in range(Config.PRETRAIN_EPOCHS):
        for index, board_position in enumerate(boards):
            if (index + 1) % Config.minibatch_size != 0:
                try:
                    value, policy, board = board_position
                except:
                    pass

                targets_pol_batch.append(policy)
                targets_val_batch.append(value)
                print(index)
                feature_batch.append(board_to_feature(board))

            else:
                feature_batch = torch.FloatTensor(feature_batch)
                targets_val_batch = Variable(torch.FloatTensor(targets_val_batch))
                targets_pol_batch = Variable(torch.FloatTensor(targets_pol_batch))
                do_backprop(feature_batch, targets_val_batch, targets_pol_batch, model, iters)
                iters = iters + 1
                feature_batch = []
                targets_val_batch = []
                targets_pol_batch = []
        print("Completed batch {} of {}".format(batch, Config.PRETRAIN_EPOCHS))


def do_backprop(batch_features, targets_val, targets_pol, model, iters):
    criterion1 = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    nn_policy_out, nn_val_out = model(batch_features)

    loss1 = criterion1(nn_val_out, targets_val)
    loss2 = cross_entropy(nn_policy_out, targets_pol)

    l2_reg = None
    for weight in model.parameters():
        if l2_reg is None:
            l2_reg = weight.norm(2)
        else:
            l2_reg = l2_reg + weight.norm(2)
    loss3 = 0.1 * l2_reg

    loss = loss1.float() - loss2.float() + loss3.float()
    print(iters)
    info = {
        'full_pt_loss1': loss1.data[0],
        'full_pt_loss2': loss2.data[0],
        'full_pt_loss3': loss3.data[0]
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, iters)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    save_trained(model, iters)

with open('labeled_boards' , 'rb') as f:
    boards = pickle.load(f)


pretrain(model,boards)
