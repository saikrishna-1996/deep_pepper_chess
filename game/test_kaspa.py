#from os import listdir
#from os.path import isfile, join
import pickle
#import argparse
#import os

labeled_board=[]
index = 0

f = '/u/gottipav/labeled_boards'
labeled_board = (pickle.load(f))

boards_data=[]
for boards in labeled_board:
    for sub_boards in boards:
        boards_data.append(sub_boards)
#with open(player_name+'_labels', 'wb') as f:
pickle.dump(boards_data, f)
