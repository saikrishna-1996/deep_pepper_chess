import numpy as np
from MCTS import MCTS
import leaf
from features import BoardToFeature
import config
from chess_env import ChessEnv


def Match_score(state):
    # after game_over get the score from white player perspective
    return

def resignation(state):
    return #True or False, the final score( +1 0 -1 )

def Game_over(env):
    # check if the list of legal move is empty or the repititions exceeded 3
    if not Board(state).legal_move:
        return True Match_score(state)
    if repetition >=3:
        return True, 0
    if resignation(state)[0]:
        return True resignation(state)[1]
    
    return False, None



def Generating_games(NUMBER_GAMES: int,env: ChessEnv):
    triplet=[]
    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 1
        while not Game_over(state, env.repetition)[0]:
            state = env.board
            step_game +=1
            if step_game == 50:
                temperature = 10e-6
            pi = MCTS(state, init_W= np.zeros((4096,)),
                       init_N = np.ones((4096,)), 
                       temp = temperature, explore_factor = 2,
                       alpha_prob,alpha_eval,dirichlet_alpha)
                        
            action_index = np.argmax(pi)
            triplet.append([state,pi])

            state = env.step( config.INDEXTOMOVE[action_index])

        z = Game_over(state, env.repetition)[1]#from white perspective

        for i in range(len(triplet)-step_game, len(triplet)):
            triplet[i].append( z )
        if game_number % 500 ==0:
            np.save('500_triplet', np.array(triplet))
            triplet = []