import numpy as np
import MCTS
import leaf
from features import BoardToFeature
from config import ALLMOVESMAP
from chess import Board


def Match_score(state):
    # after game_over get the score from white player perspective
    return

def Game_over(state, repitions):
    # check if the list of legal move is empty or the repititions exceeded 3
    if not Board(state).legal_move or repitions >=3:
        return True
    return False


def Generating_games(NUMBER_GAMES,start_state):

    state = start_state

    triplet=[]
    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 1
        while not Game_over(state):
            step_game +=1
            if step_game ==50:
                temperature = 10e-6
            pi = MCSTS(state, init_W= 0, init_N = 1, temp = temperature, explore_factor = 2)# what is the shape of this pi ????????
            action_index = np.argmax(pi)
            legal_move = Legal_move(state)
            triplet.append([state,pi])

            state = Board_render( state, legal_move[action_index])#should be able to give the same state even if no room for legal move

        z = Match_score(state)#from white perspective

        for i in range(len(triplet)-step_game, len(triplet)):
            triplet[i].append( z*(-1)**i )
        if game_number % 500 ==0:
            np.save('500_triplet', np.array(triplet))
            triplet = []
    return triplet
