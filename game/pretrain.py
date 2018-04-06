from random import shuffle

import chess.pgn
import chess.uci
# import random
import numpy as np
import torch
from torch.optim import optimizer

from config import Config
from game.features import board_to_feature
from game.stockfish import Stockfish

think_time = 10  # 1 seconds
batch_size = 32


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
    shuffled_board_positions = shuffle(board_positions)
    print("We have {} board positions".format(len(board_positions)))
    stockfish = Stockfish()

    for batch in range(Config.PRETRAIN_BATCHES):
        for index, board_position in enumerate(board_positions):
            if (index + 1) % batch_size != 0:
                feature_batch.append(board_to_feature(board_position))
                targets_batch.append(stockfish.stockfish_eval(board_position, 10))
            else:
                feature_batch = torch.from_numpy(np.asarray(feature_batch, dtype=float))
                targets_batch = torch.from_numpy(np.asarray(targets_batch, dtype=float))
                do_backprop(feature_batch, np.asarray(targets_batch), model)
                feature_batch = []
                targets_batch = []


def do_backprop(batch_features, values, model):
    criterion1 = torch.nn.MSELoss(size_average=False)
    nn_policy_out, nn_val_out = model(batch_features)
    loss = criterion1(values, nn_val_out)
    # loss2 = criterion2(policy, nn_policy_out)
    # l2_reg = None
    # for weight in model.parameters():
    #    if l2_reg is None:
    #        l2_reg = weight.norm(2)
    #    else:
    #        l2_reg = l2_reg + weight.norm(2)
    # loss3 = 0.1 * l2_reg
    # loss = loss1 + loss2 + loss3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
