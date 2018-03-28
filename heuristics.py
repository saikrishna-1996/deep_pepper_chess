from chess.uci import InfoHandler, popen_engine
import chess
import chess.polyglot
import platform


evaltime = 500  # 0.5 seconds


def play_opening(n):
    # n is the number of half moves
    board = chess.Board()
    with chess.polyglot.open_reader("./komodo_bin/strong/komodo.bin") as reader:
        for i in range(n):
            my_move = reader.choice(board).move
            board.push(my_move)
        return board


class Stockfish(object):
    def __init__(self):
        if platform.system() == 'Darwin':
            self.engine = popen_engine("./stockfish-osx")
        elif platform.system() == 'Linux':
            self.engine = popen_engine("./stockfish-linux")
        elif platform.system() == 'Windows':
            self.engine = popen_engine("./stockfish-windows")

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

    def check_resignation(self, state):
        evaluation = self.stockfish_eval(state, t=0.5)
        if abs(evaluation) > 6.5:
            return True, evaluation / abs(evaluation)
        return False, None
