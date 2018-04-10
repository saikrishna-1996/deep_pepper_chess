from random import shuffle

import chess.pgn
import chess.uci
# import random
import torch
from torch.autograd import Variable

from config import Config
from game.features import board_to_feature
from game.stockfish import Stockfish


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
    feature_batch = []
    targets_batch = []
    board_positions = get_board_position()
    shuffle(board_positions)
    print("Pretraining on {} board positions...".format(len(board_positions)))
    stockfish = Stockfish()

    for batch in range(Config.PRETRAIN_EPOCHS):
        for index, board_position in enumerate(board_positions):
            if (index + 1) % Config.minibatch_size != 0:
                feature_batch.append(board_to_feature(board_position))
                targets_batch.append(stockfish.stockfish_eval(board_position, 10))
            else:
                feature_batch = torch.FloatTensor(feature_batch)
                targets_batch = Variable(torch.FloatTensor(targets_batch))
                do_backprop(feature_batch, targets_batch, model)
                feature_batch = []
                targets_batch = []
        print("Completed batch {} of {}".format(batch, Config.PRETRAIN_EPOCHS))


def do_backprop(batch_features, targets, model):
    criterion1 = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    nn_policy_out, nn_val_out = model(batch_features)
    loss = criterion1(nn_val_out, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
