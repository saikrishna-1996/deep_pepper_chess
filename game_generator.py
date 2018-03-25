import numpy as np

from MCTS import MCTS
# this is hypothetical functions and classes that should be created by teamates.
from chess_env import ChessEnv
from config import Config
from policy_network import PolicyValNetwork_Giraffe
from features import BoardToFeature

def generate_game(model: PolicyValNetwork_Giraffe):
    triplet = []
    step_game = 0
    temperature = 1
    env = ChessEnv()
    env.reset()
    game_over = False
    while not game_over:
        step_game += 1
        if step_game == 50:
            temperature = 10e-6
        pi = MCTS(env, temp=temperature, network=model)

        action_index = np.argmax(pi)
        feature = BoardToFeature(env.board)
        triplet.append([feature, pi])
        print('')
        print(env.board)
        env.step(Config.INDEXTOMOVE[action_index])
        game_over, z = env.game_over()

    for i in range(len(triplet) - step_game, len(triplet)):
        triplet[i].append(z)

    return triplet
