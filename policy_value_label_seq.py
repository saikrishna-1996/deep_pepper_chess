import chess.pgn
import numpy as np

from config import Config
from game.chess_env import ChessEnv
from game.stockfish import Stockfish


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def evaluate_state(board):
    # print(fen)
    env = ChessEnv(board=board)
    # env.step(move)
    game_over, score = env.is_game_over()
    if game_over:
        return score
    value = env.stockfish.stockfish_eval(env.board, timeout=100)
    return value


def value_policy(board: chess.Board):
    env = ChessEnv(board)
    game_over, score = env.is_game_over()
    if game_over:
        return score, []
    stockfish = Stockfish()
    value = stockfish.stockfish_eval(env.board, timeout=100)
    next_states = []
    for move in env.board.legal_moves:
        board_copy = env.board.copy()
        board_copy.push(move)
        next_states.append(board_copy)

    actions_value = []
    for state in next_states:
        actions_value.append(evaluate_state(state))

    policy = softmax(actions_value)

    index_list = [Config.MOVETOINDEX[move.uci()] for move in env.board.legal_moves]
    map = np.zeros((5120,))
    for index, pi in zip(index_list, policy):
        map[index] = pi
    assert policy.sum() > 0.999
    return value, map
