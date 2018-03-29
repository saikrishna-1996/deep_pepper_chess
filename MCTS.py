import numpy as np
import torch

from chess_env import ChessEnv
from config import Config
from features import BoardToFeature
# this is hypothetical functions and classes that should be created by teamates.
from policy_network import PolicyValNetwork_Giraffe


def evaluate_p(list_board, network):
    list_board = [BoardToFeature(list_board[i]) for i in range(len(list_board))]
    tensor = torch.from_numpy(np.array(list_board))
    # expect that neural net ouput is a vector of probability
    probability = network.forward(tensor)[0]
    return probability.data.numpy()


def state_visited(state_list, state):
    if not state_list:
        return False, None
    for i in range(len(state_list)):
        if np.array_equal(state_list[i].env.board, state):
            return True, i
    return False, None


def Q(N, W):
    return W / float(N)


class Leaf(object):
    # This class inherit the Board class which control the board representation,
    # find legale move and next board represenation.
    # It has the ability to store and update for each leaf the
    #  number of state-action N(s,a), Q(s,a) and P(s,a)
    def __init__(self, env: ChessEnv, init_W, init_N, init_P, explore_factor):
        assert init_N.shape == (Config.d_out,)
        assert init_W.shape == (Config.d_out,)
        assert init_P.shape == (Config.d_out,)
        self.env = env
        self.P = init_P
        self.N = init_N
        self.W = init_W
        self.explore_factor = explore_factor
        self.legal_move_inds = []
        self.legal_moves = []
        self.taken_action = None
        legal_moves = env.board.legal_moves
        for move in legal_moves:
            legal_move_uci = move.uci()
            ind = Config.MOVETOINDEX[legal_move_uci]
            self.legal_moves.append(legal_move_uci)
            self.legal_move_inds.append(ind)

    @property
    def Q(self):
        return np.divide(self.W, self.N)

    @property
    def U(self):
        return np.multiply(np.multiply(self.explore_factor, self.P),
                           np.divide(np.sqrt(np.sum(self.N)), (np.add(1., self.N))))

    def best_action(self, act=False):
        if not self.env.white_to_move:
            all_moves = (np.add(self.U, -self.Q))
        else:
            all_moves = (np.add(self.U, self.Q))
        # print('MOVE IND:  '+ repr(np.argmax(all_moves[self.legal_move_inds])))

        max_list = np.argwhere(all_moves[self.legal_move_inds] == np.amax(all_moves[self.legal_move_inds]))

        move = self.legal_moves[np.random.choice(max_list.flatten(), 1)[0]]
        if act:
            self.taken_action = move
        return move

    @property
    def next_board(self):
        best_index = self.best_action
        mymove = Config.INDEXTOMOVE[best_index]
        self.env.step(mymove)
        return self.env.board
        # return self.render_action(self.board, self.best_action)#assuming the function you did
        # Do chess.Move()

    def N_update(self, action_index):
        self.N[action_index] += 1

    def W_update(self, V_next, action_index):
        self.W[action_index] += V_next

    def P_update(self, new_P):
        self.P = new_P


def legal_mask(board, all_move_probs, dirichlet=False, epsilon=None) -> np.array:
    legal_moves = board.legal_moves
    mask = np.zeros_like(all_move_probs)
    total_p = 0
    inds = []
    for legal_move in legal_moves:
        legal_move_uci = legal_move.uci()
        ind = Config.MOVETOINDEX[legal_move_uci]
        mask[ind] = 1
        inds.append(ind)
        total_p += all_move_probs[ind]

    legal_moves_prob = np.multiply(mask, all_move_probs)

    legal_moves_prob = np.divide(legal_moves_prob, total_p)

    if dirichlet:
        num_legal_moves = sum(mask)
        z = Config.D_ALPHA * np.ones(legal_moves_prob[inds].shape)
        d_noise = np.random.dirichlet(Config.D_ALPHA * np.ones(legal_moves_prob[inds].shape))
        legal_moves_prob[inds] = np.add(np.multiply((1 - epsilon), legal_moves_prob[inds]),
                                        np.multiply(epsilon, np.add(legal_moves_prob[inds], d_noise)))
        p_tot = np.sum(legal_moves_prob)
        legal_moves_prob[inds] = np.divide(legal_moves_prob[inds], p_tot)
    return legal_moves_prob


# state type and shape does not matter

def MCTS(env: ChessEnv,
         temp: float,
         network: PolicyValNetwork_Giraffe,
         explore_factor=Config.EXPLORE_FACTOR,
         dirichlet_alpha=Config.D_ALPHA,
         epsilon: float = Config.EPS,
         batch_size: int = Config.BATCH_SIZE,
         init_W=np.zeros((Config.d_out,)),
         init_N=np.ones((Config.d_out,)) * 0.001,
         init_P=np.ones((Config.d_out,)) * (1 / Config.d_out)) -> np.ndarray:
    # we can add here all our hyper-parameters
    """
    Monte-Carlo tree search function corresponds to the simulation step in the alpha_zero algorithm
    arguments: state: the root state from where the stimulation start. A board.

    :param env:
    :param temp: temperature constant for the optimum policy to control the level of exploration/
    :param network: policy network for evaluation
    :param explore_factor: hyper parameter to tune the exploration range in UCT
    :param dirichlet_alpha: alpha parameter for the dirichlet process
    :param epsilon: parameter for exploration using dirichlet noise
    :param batch_size:
    :param init_W:
    :param init_N:
    :param init_P:
    :return: return: pi: vector of policy(action) with the same shape of legale move. Shape: 4096x1
    """

    # history of leafs for all previous runs
    # env_copy = env.copy()
    leafs = []
    for simulation in range(Config.NUM_SIMULATIONS):
        curr_env = env.copy()
        state_action_list = []  # list of leafs in the same run
        moves = 0
        resign = False

        ########################
        ######## Select ########
        ########################
        game_over, v = curr_env.is_game_over(moves)
        while not game_over and not resign:
            visited, index = state_visited(leafs, curr_env.board)
            if visited:
                state_action_list.append(leafs[index])
            else:  # if state unvisited get legal moves probabilities using policy network
                if len(leafs) == 0:
                    root = Leaf(curr_env.copy(), init_W.copy(), init_N.copy(), init_P.copy(), explore_factor)
                    all_move_probs = init_P
                    legal_move_probs = legal_mask(curr_env.board, all_move_probs, dirichlet=True, epsilon=epsilon)
                    root.P = legal_move_probs
                    state_action_list.append(root)
                else:
                    all_move_probs = init_P
                    legal_move_probs = legal_mask(curr_env.board, all_move_probs)
                    state_action_list.append(
                        Leaf(curr_env.copy(), init_W.copy(), init_N.copy(), legal_move_probs.copy(), explore_factor))
                leafs.append(state_action_list[-1])

            best_action = state_action_list[-1].best_action(True)
            best_action_index = Config.MOVETOINDEX[best_action]
            # print("Best Action: " + repr(best_action))
            # print(curr_env.board)
            curr_env.step(best_action)
            moves += 1
            game_over, v = curr_env.is_game_over(moves)

        ##########################
        ### Expand and evaluate###
        ##########################

        # game_over_check, end_score = curr_env.game_over()
        # if not game_over_check:
        #    resign_check, resign_score = resignation(stockfish, curr_env.board)
        #    print ("Resignation?" + str(resign_check))
        # v = 0
        # if game_over_check:
        #    v = end_score
        # elif resign_check:
        #    v = resign_score
        # else:
        #    raise Exception("This should never happen!")

        number_batches = max(len(state_action_list) // batch_size, 1)
        start = 0
        end = min(batch_size, len(state_action_list))
        for batch in range(number_batches):
            list_p = evaluate_p([state_action_list[i].env.board for i in range(start, end)], network).data
            list_p = np.exp(list_p)
            for i in range(start, end):
                legal_move_probs = legal_mask(state_action_list[i].env.board, list_p[i - start])
                state_action_list[i].P_update(legal_move_probs)
            start = end
            end += min(batch_size, len(state_action_list) - start)

        ###############
        ### Back-up ###
        ###############

        for i in list(reversed(range(len(state_action_list)))):
            action = state_action_list[i].taken_action  # always legal since best_action
            action_index = Config.MOVETOINDEX[action]
            state_action_list[i].N_update(action_index)
            state_action_list[i].W_update(v, action_index)

        print("Simulation episode: " + str(simulation))

    N = leafs[0].N

    norm_factor = np.sum(np.power(N, temp))
    # optimum policy
    pi = np.divide(np.power(N, temp), norm_factor)

    return pi
