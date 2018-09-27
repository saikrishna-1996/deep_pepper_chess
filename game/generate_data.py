from random import shuffle

import chess.pgn
import chess.uci
# import random
import torch
from torch.autograd import Variable

from config import Config
from features import board_to_feature
from stockfish import Stockfish
import numpy as np
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('-pgn', '--pgn', help="path to pgn file", type=str, default='/home/sai/deep_pepper_chess/game/kasparov.pgn')
args = parser.parse_args()

class generate_data(object):
    def __init__(self):
        self.pgn_string = args.pgn
        self.get_groundtruth()

    def get_board_position(self):
        pgn_database = open(self.pgn_string)
        board_positions = []
        try:
            while True:
                kasgame = chess.pgn.read_game(pgn_database)
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


    def get_groundtruth(self):
        feature_batch = []
        targets_batch = []
        board_positions = self.get_board_position()
        shuffle(board_positions)
        print("done shuffling")
        print("generating evaluations on {} board positions...".format(len(board_positions)))
        # stockfish = Stockfish()

        
        for index, board_position in enumerate(board_positions):
        	print(index)
        	stockfish = Stockfish()
        	feature_batch.append(board_to_feature(board_position))
        	targets_batch.append(stockfish.stockfish_eval(board_position, 10))
        	stockfish.kill_me()
        feature_arr = np.asarray(feature_batch)
        targets_arr = np.asarray(targets_batch)
        np.save('features.txt', feature_arr)
        np.save('values.txt', targets_arr)

generate_data()