import os
import platform

import chess


def make_move_maps():
    MOVETOINDEX = {}
    INDEXTOMOVE = []
    k = 0
    for i in range(0, 64):
        for j in range(0, 64):
            move = chess.Move(i, j).uci()

            if move[3] == '8' and move[1] == '7':
                for letter in 'QRBNqrbn':
                    new_move = move + letter
                    MOVETOINDEX[new_move] = k
                    INDEXTOMOVE.append(new_move)
                    k += 1
            elif move[3] == '1' and move[1] == '2':
                for letter in 'qrbnQRBN':
                    new_move = move + letter
                    MOVETOINDEX[new_move] = k
                    INDEXTOMOVE.append(new_move)
                    k += 1
            MOVETOINDEX[move] = k
            INDEXTOMOVE.append(move)
            k += 1
    INDEXTOMOVE[0] = 'a1a1'

    return MOVETOINDEX, INDEXTOMOVE


def make_square_map():
    ALLSQUARES = {}
    files = 'abcdefgh'
    k = 0
    for j in range(1, 9):
        for i in range(8):
            square_string = str(files[i] + repr(j))
            ALLSQUARES[square_string] = k
            k += 1
    return ALLSQUARES


class Config(object):
    default_workers = 4 if platform.system() != 'Linux' else 10
    think_time = 10  # 1 seconds
    minibatch_size = 32
    PRETRAIN_EPOCHS = 1
    SQUAREMAP = {'a1': 0, 'b1': 1, 'c1': 2, 'd1': 3, 'e1': 4, 'f1': 5, 'g1': 6, 'h1': 7, 'a2': 8, 'b2': 9, 'c2': 10,
                 'd2': 11,
                 'e2': 12, 'f2': 13, 'g2': 14, 'h2': 15, 'a3': 16, 'b3': 17, 'c3': 18, 'd3': 19, 'e3': 20, 'f3': 21,
                 'g3': 22,
                 'h3': 23, 'a4': 24, 'b4': 25, 'c4': 26, 'd4': 27, 'e4': 28, 'f4': 29, 'g4': 30, 'h4': 31, 'a5': 32,
                 'b5': 33,
                 'c5': 34, 'd5': 35, 'e5': 36, 'f5': 37, 'g5': 38, 'h5': 39, 'a6': 40, 'b6': 41, 'c6': 42, 'd6': 43,
                 'e6': 44,
                 'f6': 45, 'g6': 46, 'h6': 47, 'a7': 48, 'b7': 49, 'c7': 50, 'd7': 51, 'e7': 52, 'f7': 53, 'g7': 54,
                 'h7': 55,
                 'a8': 56, 'b8': 57, 'c8': 58, 'd8': 59, 'e8': 60, 'f8': 61, 'g8': 62, 'h8': 63}

    MOVETOINDEX, INDEXTOMOVE = make_move_maps()

    GAME_SCORE = 20
    # MCTS

    RESIGN_CHECK_MIN = 40
    RESIGN_CHECK_FREQ = 10
    NUM_SIMULATIONS = 10
    SF_EVAL_THRESHOLD = 6.5
    BATCH_SIZE = default_workers
    D_ALPHA = 0.4
    EPS = 0.1
    EXPLORE_FACTOR = 2

    # Game Generator
    TEMP_REDUCE_STEP = 20
    MINGAMES = 10

    # Self Challenge
    NUM_GAMES = 4

    # PATHS
    ROOTDIR = '/u/gottipav/deep_pepper_chess'  ##### DEFINE AS REQ'd
    GAMEPATH = os.path.join(ROOTDIR, 'saved_games')
    NETPATH = os.path.join(ROOTDIR, 'saved_nets')
    BESTNET_NAME = 'BestNetwork.pth.tar'  # Example seen here: https://github.com/pytorch/examples/blob/0984955bb8525452d1c0e4d14499756eae76755b/imagenet/main.py#L139-L145

    minibatch_size = 100

    # NETWORK INFO
    d_in = 363
    h1 = 1024  # neurons in first hidden layer
    h2 = 2048  # neurons in second hidden layer
    h2p = 2048  # neurons in second hidden layer of policy network
    h2e = 512  # neurons in second hidden layer of evaluation network
    d_out = 5120  # without including under promotions. otherwise we have to increase

    # splitting the giraffe's feature vector to be input to the network
    global_features = 17
    ## the following constitute the global features:
    # side to move = 1
    # castling rights = 4
    # material configuration = 12
    piece_centric = 218
    ## the following constitute the piece-centric features:
    # piece lists with their properties = 2*(1+1+2+2+2+8)*5 = 160
    # sldiing pieces mobility = 2*(8+4+4+4+4) = 48
    # And, I just added extra 10 because, otherwise they are not adding up to 363. Someone pls recheck this.
    square_centric = 128
    ## the following constitute the square-centric features:
    # attack map = 64
    # defend map = 64
    h1a = 32  # no.of first set of neurons in first hidden layer
    h1b = 512  # no.of second set of neurons in first hidden layer
    h1c = 480  # no.of third set of neurons in first hidden layer
