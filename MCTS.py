import numpy as np
#this is hypothetical functions and classes that should be created by teamates.
import chess.uci
from policy_network import PolicyValNetwork_Full
import value_network
from chess_env import ChessEnv
from heuristics import stockfish_eval
from features import BoardToFeature
import config

def evaluate_p(list_board,network):
    list_board = [BoardToFeature(list_board[i]) for i in range(len(list_board))]
    tensor = np.array(list_board)
    #expect that neural net ouput is a vector of probability
    return network.forward(tensor)[0]

def resignation(state):
    stockfishEval = stockfish_eval(state, t=0.5)
    if abs(stockfishEval) > config.SF_EVAL_THRESHOLD:
        return True, stockfishEval / abs(stockfishEval)
    return False, None

def state_visited(state_list,state):
    if not state_list:
        return False, None
    for i in range(len(state_list)):
        if  np.array_equal(state_list[i].board, state):
            return True, i
    return False, None

def Q(N,W):
    return W/float(N)

class Leaf(object):

    #This class inherit the Board class which control the board representation,
    #find legale move and next board represenation.
    #It has the ability to store and update for each leaf the
    #  number of state-action N(s,a), Q(s,a) and P(s,a)
    def __init__(self, env:ChessEnv, init_W, init_N,init_P, explore_factor):
            assert init_N.shape == (4096,)
            assert init_W.shape == (4096,)
            assert init_P.shape == (4096,)
            self.env = env
            self.P = init_P
            self.N = init_N 
            self.W = init_W
            self.explore_factor = explore_factor

    @property
    def Q(self):
        return np.divide(self.W,self.N)

    @property
    def U(self):
        return np.multiply( np.multiply( self.explore_factor , self.P) , np.divide( np.sqrt(np.sum(self.N)),(np.add(1., self.N))))

    def best_action(self):
        index = np.argmax(np.add(self.U, self.Q)) #U and Q are lists of dimensionality no.of legal moves
        # it is nice to decorate the legal move method with property
        return index

    @property
    def next_board(self):
        best_index = self.best_action
        mymove = config.INDEXTOMOVE[best_index]
        self.env.step(mymove)
        return self.env.board
        #return self.render_action(self.board, self.best_action)#assuming the function you did
        #Do chess.Move()

    def N_update(self,action_index):
        self.N[action_index]+=1

    def W_update(self, V_next, action_index):
        self.W[action_index]+=V_next

    def P_update(self, new_P):
        self.P = new_P

def legal_mask(board,all_move_probs):
    legal_moves = board.legal_moves
    mask = np.zeros_like(all_move_probs)
    total_p = 0
    for legal_move in legal_moves:
        legal_move_uci = legal_move.uci()
        ind = config.MOVETOINDEX[legal_move_uci]
        mask[ind] = 1
        all_move_probs += 1e-6
        total_p += all_move_probs[ind]
    
    legal_moves_prob =  np.multiply(mask,all_move_probs) 

    legal_moves_prob = np.divide(legal_moves_prob,total_p)
    return legal_moves_prob

#state type and shape does not matter

def MCTS(env: ChessEnv, init_W, init_P,  init_N, explore_factor,temp,network: PolicyValNetwork_Full,dirichlet_alpha, epsilon):#we can add here all our hyper-parameters
    # Monte-Carlo tree search function corresponds to the simulation step in the alpha_zero algorithm
    # arguments: state: the root state from where the stimulation start. A board.
    #             explore_factor: hyper parameter to tune the exploration range in UCT
    #             temp: temperature constant for the optimum policy to control the level of exploration/
    #             in the Play policy
    #             optional : dirichlet noise
    #             network: policy network for evaluation
    #             dirichlet_alpha: alpha parameter for the dirichlet process
    #             epsilon : parameter for exploration using dirichlet noise
    
    # return: pi: vector of policy(action) with the same shape of legale move. Shape: 4096x1

    BATCH_SIZE = config.BATCH_SIZE
    

    #history of leafs for all previous runs
    env_copy = env.copy()
    leafs=[]
    for simulation in range (config.NUM_SIMULATIONS):
        curr_env = env.copy()
        state_action_list=[] #list of leafs in the same run
        moves = 0
        resign = False

        ########################
        ######## Select ########
        ########################

        while not curr_env.game_over()[0] and not resign:

            moves += 0.5
            if moves > config.RESIGN_CHECK_MIN and not moves % config.RESIGN_CHECK_FREQ:
                resign = resignation(curr_env.board)[0]
            
            visited, index = state_visited(leafs,curr_env.board)
            if visited:
                state_action_list.append(leafs[index])
            else: # if state unvisited get legal moves probabilities using policy network
                state_action_list.append(Leaf(curr_env, init_W, init_P, init_N, explore_factor))
                leafs.append(state_action_list[-1])
            #if leafs length is exactly 1 this mean we are in the root state then we should appy the dirichlet noise
            #(check alphago zero paper page 24)
            if len(leafs) == 1:
                leafs[0].P = np.add(np.multiply((1 - epsilon),leafs[0].P), np.multiply(epsilon, np.random.dirichlet(dirichlet_alpha, len(leaf[0].P))))
            best_action = config.INDEXTOMOVE[leafs[-1].best_action]
            curr_env = curr_env.step(best_action)

        ##########################
        ### Expand and evaluate###
        ##########################

        game_over_check, end_score = curr_env.game_over()
        resign_check, resign_score = resignation(curr_env.board)

        if game_over_check:
            v = end_score
        elif resign_check:
            v = resign_score

        number_batches = max(len(state_action_list) // BATCH_SIZE, 1)
        start = 0
        end = BATCH_SIZE
        for batch in range(number_batches):
            list_p = evaluate_p([state_action_list[i].board.env for i in range(start,end)], network)
            for i in range(start, end):
                legal_move_probs = legal_mask(state_action_list[i].env.board, list_p[i-start])
                state_action_list[i].P_update(legal_move_probs)
            start = end
            end += min(BATCH_SIZE, len(state_action_list) - start)

        ###############
        ### Back-up ###
        ###############

        for i in list(reversed(range(len(state_action_list)))):
            action_index = state_action_list[i].best_action #always legal since best_action
            state_action_list[i].N_update(action_index)
            state_action_list[i].W_update(v, action_index)
 
    N = leafs[0].N

    norm_factor = np.sum(np.power(N, temp))
    #optimum policy
    pi = np.divide(np.power(N, temp), norm_factor)

    return pi