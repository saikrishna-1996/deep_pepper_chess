import chess
import chess.pgn
import chess.uci
import policy_networks
from policy_networks import PolicyValNetwork_Giraffe

pgn = open("/u/gottipav/kasparov.pgn")
engine = chess.uci.popen_engine("/u/gottipav/stockfish-9-linux/src/stockfish")
think_time = 1000 #1 seconds
batch_size = 128
d_in = 353
h1 = 1024
h1a = 32
h1b = 512
h1c = 480
h2 = 2048
h2p = 2048
h2e = 512
d_out = 4096
global_features = 17
piece_centric = 208
square_centric = 128
handler = chess.uci.InfoHandler()
engine.info_handlers.append(handler)

x = Variable(torch.randn(batch_size, d_in))
y = Variable(torch.randn(batch_size, 1), requires_grad=False)
criticmodel = PolicyValNetworkFull

criterion = torch.nn.MSELoss(size_average=False)

for i in range(5):
    kasgame = chess.pgn.read_game(pgn)
    board = kasgame.board()
    for move in kasgame.main_line():
        cunt = 0
        while(cunt < batch_size):
            board.push(move)
            savepos[cunt] = board.copy()
            print(move)
            engine.position(board)
            evaluation = engine.go(movetime = think_time)
            eval_val[cunt] = handler.info["score"][1].cp/100.0
            cunt = cunt + 1
            if cunt == batch_size:
                cunt = 0
                do_your_shit(savepos, eval_val)
            print(eval_val)
