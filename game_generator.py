import numpy as np

from MCTS import MCTS
# this is hypothetical functions and classes that should be created by teamates.
from chess_env import ChessEnv
from config import Config


def generate_games(model, NUMBER_GAMES: int, env: ChessEnv):
    triplet = []
    for game_number in range(NUMBER_GAMES):
        step_game = 0
        temperature = 1
        env.reset()
        while not env.game_over()[0]:
            state = env.board
            step_game += 1
            if step_game == 50:
                temperature = 10e-6
            pi = MCTS(state, temp=temperature, network=model)

            action_index = np.argmax(pi)
            triplet.append([state, pi])

            env.step(Config.INDEXTOMOVE[action_index])

        z = env.game_over()[1]  # from white perspective

        for i in range(len(triplet) - step_game, len(triplet)):
            triplet[i].append(z)
        if game_number % 500 == 0:
            np.save('500_triplet', np.array(triplet))
            triplet = []
