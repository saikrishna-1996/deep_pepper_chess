import numpy as np

#this is hypothetical functions and classes that should be created by teamates.
import chess.uci
from policy_network import PolicyValNetwork_Full
import value_network
from chess_env import ChessEnv
import stockfish_eval
from features import BoardToFeature
import config


def game_over(state):
    if chess.is_game_over(state):

        score = chess.results(state)
        if score == 0:
            return True, -1
        if score == 0.5:
            return True, 0
        if score == 0:
            return True, -1

    else:
        return False, None



def Generating_games(NUMBER_GAMES: int,env: ChessEnv):
    triplet=[]
    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 1
        while not game_over(state)[0]:
            state = env.board
            step_game +=1
            if step_game == 50:
                temperature = 10e-6
            pi = MCTS(state,
                      init_W=np.zeros((4096,)),
                      init_N=np.ones((4096,)),
                      explore_factor = 2,
                      temp=temperature,
                      network=PolicyValNetwork_Full,
                      dirichlet_alpha=0.4)

            action_index = np.argmax(pi)
            triplet.append([state,pi])

            state = env.step( config.INDEXTOMOVE[action_index])

        z = game_over(state)[1]#from white perspective

        for i in range(len(triplet)-step_game, len(triplet)):
            triplet[i].append( z )
        if game_number % 500 ==0:
            np.save('500_triplet', np.array(triplet))
            triplet = []