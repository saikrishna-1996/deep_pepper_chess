import numpy as np
import MCTS
import leaf
from features import BoardToFeature
from config import ALLMOVESMAP
from chess import Board


def Match_score(state):
    # after game_over get the score from white player perspective
    return

def resignation(state):
    return #True or False, the final score( +1 0 -1 )

def Game_over(state, repetition):
    # check if the list of legal move is empty or the repititions exceeded 3
    if not Board(state).legal_move :
        return True Match_score(state)
    if repetition >=3:
        return True, 0
    if resignation(state)[0]:
        return True resignation(state)[1]
    
    return False, None



def Generating_games(NUMBER_GAMES,start_state):

    state = start_state

    triplet=[]
    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 1
        repetition = 0
        while not Game_over(state, repetition)[0]:
            step_game +=1
            if step_game ==50:
                temperature = 10e-6
            pi = MCSTS(state, init_W= [0 for i in range(64*63)],# what is the shape of this pi ????????
                       init_N = [1 for i in range range(64*63)], 
                       temp = temperature, explore_factor = 2,alpha_prob,alpha_eval,dirichlet_alpha)
            
                                                                  
            action_index = np.argmax(pi)
            legal_move = Legal_move(state)
            triplet.append([state,pi])
            
            if len(triplet)>=3 and triplet[-1][0]== triplet[-3][0]:
                repetition +=1
            else:
                repetition = 0

            state = Board_render( state, legal_move[action_index])#should be able to give the same state even if no room for legal move

        z = Game_over(state, repetition)[1]#from white perspective

        for i in range(len(triplet)-step_game, len(triplet)):
            triplet[i].append( z*(-1)**i )
        if game_number % 500 ==0:
            np.save('500_triplet', np.array(triplet))
            triplet = []
