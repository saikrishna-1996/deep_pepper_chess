
from os import listdir
from os.path import isfile, join
import pickle
import argparse
import os
parser = argparse.ArgumentParser(description='Launcher for distributed Chess trainer')
parser.add_argument('--board-index', type=int, default=0, help='board index')
parser.add_argument('--player-file', type=str, default="Karpov.pgn", help='player name')
parser.add_argument('--load-path', type=str, default=os.path.dirname(os.path.realpath(__file__)), help='path of files')

args = parser.parse_args()
player_name= os.path.splitext(os.path.basename(args.player_file))

onlyfiles = [f for f in listdir(args.load_path) if isfile(join(args.load_path, f))]
labeled_board=[]
index = 0

for file in onlyfiles:
    with open(args.load_path+file,'rb') as f:
        labeled_board.append(pickle.load(f))
    index += 1

boards_data=[]
for boards in labeled_board:
    for sub_boards in boards:
        boards_data.append(sub_boards)
with open(player_name+'_labels', 'wb') as f:
    pickle.dump(boards_data, f)
