import chess
import chess.uci
import numpy as np

def BoardToFeature(board):

    feature = np.zeros(363)
    "START: GLOBAL FEATURES"

    #side-to-move
    feature[0] = board.turn

    #castling rights
    feature[1] = board.has_kingside_castling_rights(chess.WHITE)
    feature[2] = board.has_queenside_castling_rights(chess.WHITE)
    feature[3] = board.has_kingside_castling_rights(chess.BLACK)
    feature[4] = board.has_queenside_castling_rights(chess.BLACK)

    #material configuration
    feature[5] = len(board.pieces(chess.KING, chess.WHITE))
    feature[6] = len(board.pieces(chess.QUEEN, chess.WHITE))
    feature[7] = len(board.pieces(chess.ROOK, chess.WHITE))
    feature[8] = len(board.pieces(chess.BISHOP, chess.WHITE))
    feature[9] = len(board.pieces(chess.KNIGHT, chess.WHITE))
    feature[10] = len(board.pieces(chess.PAWN, chess.WHITE))
    feature[11] = len(board.pieces(chess.KING, chess.BLACK))
    feature[12] = len(board.pieces(chess.QUEEN, chess.BLACK))
    feature[13] = len(board.pieces(chess.ROOK, chess.BLACK))
    feature[14] = len(board.pieces(chess.BISHOP, chess.BLACK))
    feature[15] = len(board.pieces(chess.KNIGHT, chess.BLACK))
    feature[16] = len(board.pieces(chess.PAWN, chess.BLACK))

    "START: PIECE LIST"

    #white king
    feature[17] = 1
    wk_squares = board.pieces(chess.KING, chess.WHITE)



    #We should make sure that bishops are not messed up.
