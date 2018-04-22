import os
import sys
import argparse
import chess.pgn
import time
import pickle
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from network.policy_network import PolicyValNetwork_Giraffe
from logger import Logger


def get_board_position(file_name):
    pgn = open(file_name)

    board_positions = []
    try:
        while True:
            kasgame = chess.pgn.read_game(pgn)
            if kasgame is None:
                break
            board = kasgame.board()
            board_positions.append(board.copy())
            for move in kasgame.main_line():
                board.push(move)
                board_positions.append(board.copy())
        return board_positions
    except Exception:
        print("We have {} board positions".format(len(board_positions)))
        return board_positions
parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')
parser.add_argument('--player-file', type=str, default='Karpov.pgn', help='board index')
parser.add_argument('--dump-path', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='board index')

args = parser.parse_args()

player_name =  os.path.splitext(os.path.basename(args.player_file))[0]

boards = get_board_position(args.player_file)
with open(args.dump_path+'/board_'+ player_name, 'wb') as fp:
    pickle.dump(boards, fp)

