import chess.pgn
import chess.uci
from chess_env import ChessEnv
# import random
import torch
from torch.autograd import Variable

import features
from policy_network import PolicyValNetwork_Giraffe

# Not used


engine = chess.uci.popen_engine("./stockfish")
think_time = 1000  # 1 seconds
batch_size = 32

handler = chess.uci.InfoHandler()
engine.info_handlers.append(handler)

savepos = Variable(torch.randn(batch_size, d_in))
eval_val = Variable(torch.randn(batch_size, 1), requires_grad=False)
critic_model = Critic_Giraffe(d_in, gf, pc, sc, h1a, h1b, h1c, h2, 1)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(critic_model.parameters(), lr=1e-4, momentum=0.9)  # change it Adam or Adagrad may be


def pretrain_model(model=PolicyValNetwork_Giraffe(), games=None):
    if games is None:
        game_data = load_kaspagames
    else:
        game_data = games

    #for games_trained in range(


def play_pgn(model: PolicyValNetwork_Giraffe):
    pgn = open("/u/gottipav/kasparov.pgn")
    singlet = []
    for i in range(num_games):
        kasgame = chess.pgn.read_game(pgn)
        board = kasgame.board()
        board.push(move)
        singlet.append(board)


def do_your_shit(board, stock_eval):
    optimizer.zero_grad()
    critic_eval = critic_model(board)
    loss = criterion(critic_eval, stock_eval)
    print(loss.data[0])
    loss.backward()
    optimizer.step()


cunt = 0
# eval_val = []
# savepos = []
for i in range(5):
    kasgame = chess.pgn.read_game(pgn)
    board = kasgame.board()
    for move in kasgame.main_line():

        if move is None:
            kasgame = chess.pgn.read_game(pgn)
            board = kasgame.board()
        cunt = cunt + 1
        if cunt == batch_size:
            do_your_shit(savepos, eval_val)
            cunt = 0
            # eval_val = []
            # savepos = []
        else:
            board.push(move)
            features = features.BoardToFeature(board)
            savepos[cunt, :] = torch.FloatTensor(features)
            print(move)
            engine.position(board)
            evaluation = engine.go(movetime=think_time)
            # eval_val[cunt,0] =
            shit = handler.info["score"][1].cp / 100.0
            print(shit)
            print(type(shit))
            print(torch.Tensor(shit))
            # print(type(handler.info["score"][1].cp/100.0))
            # print(torch.FloatTensor(float(handler.info["score"][1].cp/100.0)))
            print(eval_val)
