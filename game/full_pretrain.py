import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0,parentPath)

from random import shuffle
import chess.pgn
import chess.uci
# import random
import torch
from torch.autograd import Variable
from config import Config
from game.features import board_to_feature
from game.stockfish import Stockfish
from logger import Logger
import parallel_mcts_test

#set the logger
logger = Logger('./logs')

def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets.doble() * torch.log(pred).double(), 1))


def get_board_position():
    pgn = open("./game/kasparov.pgn")
    board_positions = []
    try:
        while True:
            kasgame = chess.pgn.read_game(pgn)
            if kasgame is None:
                break
            board = kasgame.board()
            board_positions.append(board.copy())
            for move in kasgame.main_line():
                board.push(move)
                board_positions.append(board.copy())
    except Exception:
        print("We have {} board positions".format(len(board_positions)))
        return board_positions


def pretrain(model):
    iters = 0
    feature_batch = []
    targets_val_batch = []
    targets_pol_batch = []
    board_positions = get_board_position()
    shuffle(board_positions)
    print("Pretraining on {} board positions...".format(len(board_positions)))
    stockfish = Stockfish()

    for batch in range(Config.PRETRAIN_EPOCHS):
        for index, board_position in enumerate(board_positions):
            if (index + 1) % Config.minibatch_size != 0:
                feature_batch.append(board_to_feature(board_position))
                targets_val_batch.append(stockfish.stockfish_eval(board_position, 10))
                nvm, targets_pol_batch.append(parallel_mcts_test.value_policy(board_position))
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
        if l2.reg is None:
            l2_reg = weight.norm(2)
        else:
            l2_reg = l2_reg + weight.norm(2)
    loss3 = 0.1 * l2_reg

    loss = loss1.float() - loss2.float() + loss3.float()
    
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
