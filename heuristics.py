from chess.uci import InfoHandler, popen_engine

evaltime = 500  # 0.5 seconds


def stockfish_eval(board, t=500):
    handler = InfoHandler()
    engine = popen_engine("./stockfish")  # give the correct path here
    engine.info_handlers.append(handler)
    engine.position(board)
    evaluation = engine.go(movetime=t)
    eval_val = handler.info["score"][1].cp / 100.0
    return eval_val


def stockfish_result(board):
    eval_val = stockfish_eval(board)
    if eval_val > 6.5:
        return 1
    elif eval_val < -6.5:
        return -1
    else:
        return 0
