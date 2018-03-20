import chess
from chess.uci import InfoHandler, popen_engine

pgnfilename = str(arguments[1])

# Read pgn file (entire game, wtf)
with open(pgnfilename) as f:
    game = chess.pgn.read_game(f)

# create a chess.Board() from it in the final position
game = game.end()
board = game.board()  # in our case, we should directly have this

# PGN to FEN conversion, if required
print("FEN of the last position of the game:", board.fen())

# load your engine now
handler = InfoHandler()
engine = popen_engine('...\stockfish_8x_64')  # give the correct path here
engine.info_handlers.append(handler)

# send your position to the engine
engine.position(board)

# set your evaluation time, in ms: This is very important. change, if required
evaltime = 5000  # 5 seconds
evaluation = engine.go(movetime=evaltime)

eval_val = handler.info["score"][1].cp / 100.0

# tell me whose turn to play it is, so that I can write if-else condition to output a boolean flag
