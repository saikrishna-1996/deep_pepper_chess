import numpy as np
import MCTS
import leaf
from features import BoardToFeature
from config import ALLMOVESMAP
from chess import Board

import old_alpha
import new_alpha

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

def Generating_challenge(NUMBER_GAMES, start_state):
    state = start_state

    new_alpha_score=[]
    old_alpha_score=[]

    for game_number in range(NUMBER_GAMES):

        step_game = 0
        temperature = 10e-6

        if np.random.binomial(1, 0.5) ==1:
            white = new_alpha
            balck = old_alpha
        else:
            white = old_alpha
            black = new_alpha


        repition =0
        state_list = []

        while not Game_over(state,repetition)[0]:

            state_list.append(state)

            if len(state_list)>3 and state_list[-1] == state_list[-3]:
                repetition += 1
            else:
                #reseting the repition
                repetition = 0


            if step_game%2:

                player = black
            else:

                player = white
            step_game += 1


            pi = MCSTS(state, init_W=[0 for i in range(64 * 63)],  # what is the shape of this pi ????????
                       init_N=[1 for i in range(64 * 63)],
            temp = temperature, explore_factor = 2,alpha_prob= player ,alpha_eval= player)


            action_index = np.argmax(pi)
            legal_move = Legal_move(state)

            state = Board_render(state, legal_move[action_index])
            # should be able to give the same state even if no room for legal move

        z = Game_over(state, repetition)[1]
        # from white perspective

        if white == alpha1:
            new_alpha_score.append(+z)
            old_alpha_score.append(-z)

        else:
            new_alpha_score.append(-z)
            old_alpha_score.append(+z)



    if sum(new_alpha_score) >sum(old_alpha_score):
        winner == 'new_alpha'

    elif sum(new_alpha_score) < sum(old_alpha_score):
        winner == 'old_alpha'

    else:
        winner == None
    return alpha1_score,alpha2_score,winner
