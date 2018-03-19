import chess
import chess.uci



evaltime = 500 #0.5 seconds

def stockfish_eval(board, t=500):
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine("./stockfish") #give the correct path here
    engine.info_handlers.append(handler)
    engine.position(board)
    evaluation = engine.go(movetime = t)
    eval_val = handler.info["score"][1].cp/100.0
    return eval_val

def stockfish_result(board):
    eval_val = stockfish_eval(board)
    if abs(eval_val) > 6.5:
        return eval_val/abs(eval_val)
    else:
        return None
