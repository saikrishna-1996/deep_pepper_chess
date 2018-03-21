from chess.uci import InfoHandler, popen_engine

evaltime = 500  # 0.5 seconds


class Stockfish(object):
    def __init__(self):
        self.engine = popen_engine("./stockfish")

    def stockfish_eval(self, board, t=500):
        handler = InfoHandler()
        self.engine.info_handlers.append(handler)
        self.engine.position(board)
        evaluation = self.engine.go(movetime=t)
        if handler.info["score"][1].mate is None:
            eval_val = handler.info["score"][1].cp / 100.0
        else:
            if handler.info["score"][1].mate < 0:
                eval_val = -100.0
            else:
                eval_val = 100.0
        return eval_val

    def stockfish_result(self, board):
        eval_val = self.stockfish_eval(board)
        if eval_val > 6.5:
            return 1
        elif eval_val < -6.5:
            return -1
        else:
            return 0
