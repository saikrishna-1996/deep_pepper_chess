import os
import sys
import time
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
import pickle
import parallel_mcts_test
import argparse
import torch
import chess.pgn

def get_board_position(pgn):
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
    except Exception:
        print("We have {} board positions".format(len(board_positions)))
        return board_positions


def cross_entropy(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets.double() * torch.log(pred).double(), 1))


parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')
parser.add_argument('--board-index', type=int, default=0, help='board index')
parser.add_argument('--player-file', type=str, default="Karpov.pgn", help='player file')
parser.add_argument('--dump-path', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='path to save files')
parser.add_argument('--load-boards', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='player boards path')

args = parser.parse_args()
player_name =  os.path.splitext(os.path.basename(args.player_file))[0]

with open (args.load_boards+'/board_'+ player_name,'rb') as f:
    boards = pickle.load(f)

list=[]
for i in range(10):
    list.append(parallel_mcts_test.value_policy(boards[args.board_index+i]))

with open(args.dump_path+ player_name+str(args.board_index)+'-'+str(args.board_index+i), 'wb') as fp:
    pickle.dump(list, fp)



