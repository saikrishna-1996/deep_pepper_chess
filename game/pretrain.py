from random import shuffle

import chess.pgn
import chess.uci
# import random
import torch
from torch.optim import optimizer

from config import Config
from game.features import board_to_feature

think_time = 10  # 1 seconds
batch_size = 32


def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets.doble() * torch.log(pred).double(), 1))


def get_board_position():
    pgn = open("./kasparov.pgn")
    board_positions = []
    while True:
        kasgame = chess.pgn.read_game(pgn)
        if kasgame is None:
            break
        board = kasgame.board()
        board_positions.append(board.copy())
        for move in kasgame.main_line():
            board.push(move)
            board_positions.append(board.copy())
    return board_positions


def pretrain(model):
    training_data = []
    board_positions = get_board_position()
    board_positions = shuffle(board_positions)

    for batch in range(Config.PRETRAIN_BATCHES):
        for index in range(board_positions):
            if index % batch_size != 0:
                val = stockfish_eval(board_positions[index], 10)
                training_data.append([board_positions[index], val])
            else:
                do_backprop(training_data, model)
                training_data = []


def stockfish_pol(board_position):
    pass


def do_backprop(training_data, model):
    features, values = [(board_to_feature(training_data[i][0]), training_data[i][1]) for i in training_data]

    criterion1 = torch.nn.MSELoss(size_average=False)
    nn_policy_out, nn_val_out = model(features)
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

# def do_your_shit(board, stock_eval):
#    optimizer.zero_grad()
#    critic_eval = critic_model(board)
#    loss = criterion(critic_eval, stock_eval)
#    print(loss.data[0])
#    loss.backward()
#    optimizer.step()


# cunt = 0
## eval_val = []
## savepos = []
# for i in range(5):
#    kasgame = chess.pgn.read_game(pgn)
#    board = kasgame.board()
#    for move in kasgame.main_line():
#
#        if move is None:
#            kasgame = chess.pgn.read_game(pgn)
#            board = kasgame.board()
#        cunt = cunt + 1
#        if cunt == batch_size:
#            do_your_shit(savepos, eval_val)
#            cunt = 0
#            # eval_val = []
#            # savepos = []
#        else:
#            board.push(move)
#            features = features.BoardToFeature(board)
#            savepos[cunt, :] = torch.FloatTensor(features)
#            print(move)
#            engine.position(board)
#            evaluation = engine.go(movetime=think_time)
#            # eval_val[cunt,0] =
#            shit = handler.info["score"][1].cp / 100.0
#            print(shit)
#            print(type(shit))
#            print(torch.Tensor(shit))
#            # print(type(handler.info["score"][1].cp/100.0))
#            # print(torch.FloatTensor(float(handler.info["score"][1].cp/100.0)))
#            print(eval_val)
