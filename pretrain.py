import chess.pgn
import chess.uci
# import random
import torch

from policy_network import PolicyValNetwork_Giraffe

# Not used


#engine = chess.uci.popen_engine("./stockfish")
think_time = 1000  # 1 seconds
batch_size = 32

#handler = chess.uci.InfoHandler()
#engine.info_handlers.append(handler)

#savepos = Variable(torch.randn(batch_size, d_in))
#eval_val = Variable(torch.randn(batch_size, 1), requires_grad=False)
#critic_model = Critic_Giraffe(d_in, gf, pc, sc, h1a, h1b, h1c, h2, 1)

#criterion = torch.nn.MSELoss(size_average=False)
#optimizer = torch.optim.SGD(critic_model.parameters(), lr=1e-4, momentum=0.9)  # change it Adam or Adagrad may be


def pretrain_model(model=PolicyValNetwork_Giraffe(), games=None):
    if games is None:
        game_data = load_kaspagames
    else:
        game_data = games

    #for games_trained in range(


def play_pgn():
    pgn = open("/u/gottipav/kasparov.pgn")
    board_positions = []
    for i in range(num_games):
        kasgame = chess.pgn.read_game(pgn)
        board = kasgame.board()
        board.push(move)
        board_positions.append(board)
        return board_positions

def get_triplet_and_backprop(board_positions):
    triplet=[]
    for index in range(board_positions):
        if index % batch_size  != 0:
            val = stockfish_eval(board_positions[index], 1000)
            moves_with_probab = board_positions[index]
            triplet.append([board_positions[index], val, moves_with_probab])
        else:
            do_backprop(triplet)
            triplet = []

def stockfish_pol(board_position):



def do_backprop(features, policy, val, model):
    criterion1 = torch.nn.MSELoss(size_average=False)
    criterion2 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, momentum=0.9)
    nn_policy_out, nn_val_out = model(features)
    loss1 = criterion1(val, nn_val_out)
    loss2 = criterion2(policy, nn_policy_out)
    l2_reg = None
    for weight in model.parameters():
        if l2_reg is None:
            l2_reg = weight.norm(2)
        else:
            l2_reg = l2_reg + weight.norm(2)
    loss3 = 0.1*l2_reg
    loss = loss1 + loss2 + loss3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



#def do_your_shit(board, stock_eval):
#    optimizer.zero_grad()
#    critic_eval = critic_model(board)
#    loss = criterion(critic_eval, stock_eval)
#    print(loss.data[0])
#    loss.backward()
#    optimizer.step()


#cunt = 0
## eval_val = []
## savepos = []
#for i in range(5):
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
