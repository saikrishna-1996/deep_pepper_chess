import multiprocessing

import numpy as np

from MCTS import MCTS
from chess_env import ChessEnv
from config import Config
from features import BoardToFeature
from policy_network import PolicyValNetwork_Giraffe


def generate_game(model: PolicyValNetwork_Giraffe):
    triplets = []
    step_game = 0
    temperature = 1
    env = ChessEnv()
    env.reset()
    game_over = False
    moves = 0
    game_over, z = env.is_game_over(moves)
    while not game_over:
        moves += 1
        step_game += 1
        if step_game == 50:
            temperature = 10e-6
        pi = MCTS(env, temp=temperature, network=model)

        action_index = np.argmax(pi)
        feature = BoardToFeature(env.board)
        triplets.append([feature, pi])
        print('')
        print(env.board)
        print("Running on {} ".format(multiprocessing.current_process()))
        env.step(Config.INDEXTOMOVE[action_index])
        game_over, z = env.is_game_over(moves)

    for i in range(len(triplets) - step_game, len(triplets)):
        triplets[i].append(z)

    return triplets
