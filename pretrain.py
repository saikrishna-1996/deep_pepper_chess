import chess
import chess.pgn
import chess.uci

pgn = open("/u/gottipav/kasparov.pgn")
engine = chess.uci.popen_engine("/u/gottipav/stockfish-9-linux/src/stockfish")
think_time = 5000 #5 seconds
handler = chess.uci.InfoHandler()
engine.info_handlers.append(handler)

for i in range(5):
    kasgame = chess.pgn.read_game(pgn)
    board = kasgame.board()
    for move in kasgame.main_line():
        board.push(move)
        print(move)
        engine.position(board)
        evaluation = engine.go(movetime = think_time)
        eval_val = handler.info["score"][1].cp/100.0
        print(eval_val)
