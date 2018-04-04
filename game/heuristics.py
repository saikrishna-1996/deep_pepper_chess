import platform

from chess.uci import popen_engine, InfoHandler

from os import path

basepath = str(path.abspath(path.dirname(__file__)))

import chess

evaltime = 500  # 0.5 seconds

class Playbook(object):
    def __init__(self):
        self.reader = chess.polyglot.open_reader(basepath + "komodo_bin/strong/komodo.bin")
    def play_opening(n,env):
        # n is the number of half moves
        board = chess.Board()
        with self.reader as reader:
            try:
                for i in range(n):
                    my_move = reader.choice(env.board).move.uci()
                    env.step(my_move)
            except:
                print('Error when using playbook')
            return env, i


class Stockfish(object):
    def __init__(self):
        if platform.system() == 'Darwin':
            self.engine = popen_engine(basepath + "/stockfish-osx")
        elif platform.system() == 'Linux':
            self.engine = popen_engine(basepath + "/stockfish-linux")
        elif platform.system() == 'Windows':
            self.engine = popen_engine(basepath + "/stockfish-windows")

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
