
"""
Chess environment used for playing games. 
"""

import copy
import enum

import chess.pgn
import numpy as np

from config import Config
from game.stockfish import Stockfish

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

pieces_order = 'KQRBNPkqrbnp'
castling_order = 'KQkq'

ind = {pieces_order[i]: i for i in range(12)}

class ChessEnv:

    def __init__(self, board=None):
        self.board = board
        self.num_halfmoves = 0
        self.winner = None  # type: Winner
        self.resigned = False
        self.result = None
        self.state_count = dict()
        self.stockfish = Stockfish()

    def reset(self):
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.winner = None
        self.resigned = False

        # count first board state
        self.state_count = dict()
        transposition = self.board._transposition_key()
        self.state_count[transposition] = 1

        return self

    def update(self, board):
        self.board = chess.Board(board)
        self.winner = None
        self.resigned = False
        return self

    @property
    def done(self):
        return self.winner is not None

    @property
    def white_won(self):
        return self.winner == Winner.white

    @property
    def white_to_move(self):
        return self.board.turn == chess.WHITE

    @property
    def repetition(self):
        return self.state_count[self.board._transposition_key]

    def step(self, action: str, check_over=True):
        """
        :param action:
        :param check_over:
        :return:
        """
        if check_over and action is None:
            self._resign()
            return
        self.board.push_uci(action)
        self.update_state_count()

        self.num_halfmoves += 1

        if check_over and self.board.result(claim_draw=True) != "*":
            self._game_over()

    def _game_over(self):
        if self.winner is None:
            self.result = self.board.result(claim_draw=True)
            if self.result == '1-0':
                self.winner = Winner.white
            elif self.result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def is_game_over(self, moves=0, res_check=False, testing_flag=False) -> tuple:
        """
        :param moves number of half moves since games start
        :param res_check indicate whether to check resignation via stockfish
        :param testing_flag flag automatically ending game if true
        :return tuple where first element indicates if game is over, second element indicates end game score
        """
        if testing_flag:
            return True, 0
        if self.board.is_game_over():
            score = self.board.result()
            if score == '0-1':
                return True, -Config.GAME_SCORE
            if score == '1/2-1/2':
                return True, 0
            if score == '1-0':
                return True, Config.GAME_SCORE
        elif (moves > Config.RESIGN_CHECK_MIN) and (not moves % Config.RESIGN_CHECK_FREQ) and res_check:
            return self.stockfish.check_resignation(self.board)
        return False, None

    def _resign(self):
        self.resigned = True
        if self.white_to_move:
            self.winner = Winner.black
            self.result = "0-1"
        else:
            self.winner = Winner.white
            self.result = "1-0"

    def adjudicate(self):
        score = self.testeval(absolute=True)
        if abs(score) < 0.01:
            self.winner = Winner.draw
            self.result = "1/2-1/2"
        elif score > 0:
            self.winner = Winner.white
            self.result = "1-0"
        else:
            self.winner = Winner.black
            self.result = "0-1"

    def ending_average_game(self):
        self.winner = Winner.draw
        self.result = "1/2-1/2"

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    # returns list of legal moves in UCI format.
    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    def deltamove(self, fen_next):
        moves = list(self.board.legal_moves)
        for mov in moves:
            self.board.push(mov)
            fee = self.board.fen()
            self.board.pop()
            if fee == fen_next:
                return mov.uci()
        return None

    def replace_tags(self):
        return replace_tags_board(self.board.fen())

    def canonical_input_planes(self):
        return canon_input_planes(self.board.fen())

    def testeval(self, absolute=False) -> float:
        return testeval(self.board.fen(), absolute)

    # returns 2 planes, one for each repetition of state
    def repetition_planes(self):
        state = self.board._transposition_key()
        if self.state_count[state] == 1:
            rep1 = np.full([8, 8], 1, dtype=np.float32)
            rep2 = np.full([8, 8], 0, dtype=np.float32)

        elif self.state_count[state] == 2:
            rep1 = np.full([8, 8], 1, dtype=np.float32)
            rep2 = np.full([8, 8], 1, dtype=np.float32)

        else:
            rep1 = np.full([8, 8], 0, dtype=np.float32)
            rep2 = np.full([8, 8], 0, dtype=np.float32)
        reps = np.vstack((rep1, rep2))
        assert reps.shape == (2, 8, 8)
        return reps

    def update_state_count(self):
        state = self.board._transposition_key()
        if state in self.state_count:
            self.state_count[state] += 1
        else:
            self.state_count[state] = 1


def testeval(fen, absolute=False) -> float:
    piece_vals = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1}  # somehow it doesn't know how to keep its queen
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue
        # assert c.upper() in piece_vals
        if c.isupper():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.upper()]
            tot += piece_vals[c.upper()]
    v = ans / tot
    if not absolute and is_black_turn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3)  # arbitrary


def check_current_planes(realfen, planes):
    cur = planes[0:12]
    assert cur.shape == (12, 8, 8)
    fakefen = ["1"] * 64
    for i in range(12):
        for rank in range(8):
            for file in range(8):
                if cur[i][rank][file] == 1:
                    assert fakefen[rank * 8 + file] == '1'
                    fakefen[rank * 8 + file] = pieces_order[i]

    castling = planes[12:16]
    fiftymove = planes[16][0][0]
    ep = planes[17]

    castlingstring = ""
    for i in range(4):
        if castling[i][0][0] == 1:
            castlingstring += castling_order[i]

    if len(castlingstring) == 0:
        castlingstring = '-'

    epstr = "-"
    for rank in range(8):
        for file in range(8):
            if ep[rank][file] == 1:
                epstr = coord_to_alg((rank, file))

    realfen = maybe_flip_fen(realfen, flip=is_black_turn(realfen))
    realparts = realfen.split(' ')
    assert realparts[1] == 'w'
    assert realparts[2] == castlingstring
    assert realparts[3] == epstr
    assert int(realparts[4]) == fiftymove
    # realparts[5] is the fifty-move clock, discard that
    return "".join(fakefen) == replace_tags_board(realfen)


def canon_input_planes(fen):
    fen = maybe_flip_fen(fen, is_black_turn(fen))
    return all_input_planes(fen)


def all_input_planes(fen):
    current_aux_planes = aux_planes(fen)
    assert current_aux_planes.shape == (5, 8, 8)

    history_both = to_planes(fen)
    assert history_both.shape == (6, 8, 8)

    return history_both, current_aux_planes


def maybe_flip_fen(fen, flip=False):
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')

    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(row) for row in reversed(rows)]) \
           + " " + ('w' if foo[1] == 'b' else 'b') \
           + " " + "".join(sorted(swapall(foo[2]))) \
           + " " + foo[3] + " " + foo[4] + " " + foo[5]


def aux_planes(fen):
    foo = fen.split(' ')

    no_progress_count = int(foo[4])
    fifty_move = np.full((8, 8), no_progress_count, dtype=np.float32)
    castling = foo[2]
    auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                        np.full((8, 8), int('Q' in castling), dtype=np.float32),
                        np.full((8, 8), int('k' in castling), dtype=np.float32),
                        np.full((8, 8), int('q' in castling), dtype=np.float32),
                        fifty_move]

    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (5, 8, 8)
    return ret


# FEN board is like this:
# a8 b8 .. h8
# a7 b7 .. h7
# .. .. .. ..
# a1 b1 .. h1
# 
# FEN string is like this:
#  0  1 ..  7
#  8  9 .. 15
# .. .. .. ..
# 56 57 .. 63

# my planes are like this:
# 00 01 .. 07
# 10 11 .. 17
# .. .. .. ..
# 70 71 .. 77
#

def alg_to_coord(alg):
    rank = 8 - int(alg[1])  # 0-7
    file = ord(alg[0]) - ord('a')  # 0-7
    return rank, file


def coord_to_alg(coord):
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number


def to_planes(fen):
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8)
    return pieces_both


def replace_tags_board(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")


def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'
