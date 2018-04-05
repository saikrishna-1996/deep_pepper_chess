import sys

import numpy as np
import torch

# this is hypothetical functions and classes that should be created by teamates.
from config import Config
from game.chess_env import ChessEnv
from game.features import board_to_feature
from network.policy_network import PolicyValNetwork_Giraffe


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def evaluate_p(list_board, network):
    list_board = [board_to_feature(list_board[i]) for i in range(len(list_board))]
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


class Node(object):
    # This class inherit the Board class which control the board representation,
    # find legal move and next board represenation.
    # It has the ability to store and update for each leaf the
    #  number of state-action N(s,a), Q(s,a) and P(s,a)
    def __init__(self, env: ChessEnv, init_W, init_N, init_P, explore_factor, parent=None):
        assert init_N.shape == (Config.d_out,)
        assert init_W.shape == (Config.d_out,)
        assert init_P.shape == (Config.d_out,)

        self.env = env
        self.init_p = init_P
        self.init_n = init_N
        self.init_w = init_W
        self.P = init_P
        self.N = init_N
        self.W = init_W
        self.parent = parent
        self.children = []
        self.explore_factor = explore_factor
        self.legal_move_inds = []
        self.legal_moves = []
        self.taken_action = None
        self.best_child = None
        self.best_action = None
        self.new_action = None
        legal_moves = env.board.legal_moves
        for move in legal_moves:
            legal_move_uci = move.uci()
            ind = Config.MOVETOINDEX[legal_move_uci]
            self.legal_moves.append(legal_move_uci)
            self.legal_move_inds.append(ind)

    @property
    def Q(self):
        Q = np.divide(self.W, self.N)
        Q[np.isnan(Q)] = 0
        return Q

    @property
    def U(self):
        return np.multiply(np.multiply(self.explore_factor, self.P),
                           np.divide(np.sqrt(np.sum(self.N)), (np.add(1., self.N))))

    @property
    def best_action_update(self):
        if not self.env.white_to_move:
            all_moves = (np.add(self.U, -self.Q))
        else:
            all_moves = (np.add(self.U, self.Q))

        max_list = np.argwhere(all_moves[self.legal_move_inds] == np.amax(all_moves[self.legal_move_inds]))
        move = self.legal_moves[np.random.choice(max_list.flatten(), 1)[0]]
        self.new_action = True
        if move == self.taken_action:
            self.new_action = False

        self.taken_action = move
        return move

    @property
    def next_env(self):
        best_index = self.taken_action
        next_env = self.env.copy()
        next_env.step(best_index)
        return next_env

    def expand(self):
        # children = []
        # for action in self.env.legal_moves:
        #     next_env = self.env.copy()
        #     next_env.step(str(action))
        #     children.append(Node(next_env.copy(), self.init_w.copy(), self.init_n.copy(), self.init_p.copy(), self.explore_factor, self))
        # self.children = children
        # return
        new_child = self.env.copy()
        new_child.step(self.best_action_update)
        self.best_child = Node(new_child.copy(), self.init_w.copy(), self.init_n.copy(), self.init_p.copy(), self.explore_factor, self)
        self.children.append(self.best_child)
        return

    def best_child_update(self):

        best_child = self.env.copy()
        best_child.step(self.best_action_update)
        if self.new_action:
            # self.best_child = Node(best_child.copy(), self.init_w.copy(), self.init_n.copy(), self.init_p.copy(), self.explore_factor, self)
            # self.children.append(self.best_child)

            for child in self.children:
                if child.env.board == best_child.board:
                    self.best_child = child
                    return
            self.best_child = Node(best_child.copy(), self.init_w.copy(), self.init_n.copy(), self.init_p.copy(),
                                   self.explore_factor, self)
            self.children.append(self.best_child)
        return

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

def MCTS(env: ChessEnv, temp: float,
         network: PolicyValNetwork_Giraffe,
         dirichlet_alpha=Config.D_ALPHA,
         batch_size: int = Config.BATCH_SIZE) -> np.ndarray:
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
    # history of archive for all previous runs
    # env_copy = env.copy()
    init_W = np.zeros((Config.d_out,))
    init_N = np.zeros((Config.d_out,))
    init_P = np.ones((Config.d_out,)) * (1 / Config.d_out)

    root_node = Node(env, init_W.copy(), init_N.copy(), init_P.copy(), Config.EXPLORE_FACTOR)
    for simulation in range(Config.NUM_SIMULATIONS):
        curr_node, moves, game_over, z = select(root_node, init_W.copy(), init_N.copy(), init_P.copy())
        v, leaf = expand_and_eval(curr_node, network, game_over, z)
        backup(leaf, v)

    N = root_node.N

    norm_factor = np.sum(np.power(N, temp))
    # optimum policy
    pi = np.divide(np.power(N, temp), norm_factor)

    return pi


########################
######## Select ########
########################
# Traverses from root node to leaf node using UCB selection
def select(root_node, init_W, init_N, init_P):
    curr_node = root_node
    moves = 0
    game_over, z = curr_node.env.is_game_over(moves)
    while curr_node.children and not game_over:
        curr_node.best_child_update()
        curr_node = curr_node.best_child
        moves += 1
        # print(moves)
        game_over, z = curr_node.env.is_game_over(moves)

    # print(len(curr_node.children))
    return curr_node, moves, game_over, z


##########################
### Expand and evaluate###
##########################
# Once at a leaf node expand using the network to get it's P values and it's estimated value
def expand_and_eval(node, network, game_over, z):
    if game_over:
        return z, node
    # expand
    node.expand()
    # evaluate
    all_move_probs, v = network.forward(torch.from_numpy(board_to_feature(node.env.board)).unsqueeze(0))
    all_move_probs = all_move_probs.squeeze().data.numpy()
    if node.parent:

        legal_move_probs = legal_mask(node.env.board, all_move_probs)
    else:
        legal_move_probs = legal_mask(node.env.board, all_move_probs, dirichlet=True, epsilon=Config.EPS)
    node.P_update(legal_move_probs)
    # node.best_action_update(True)

    return v.squeeze().data.numpy(), node


###############
### Back-up ###
###############

def backup(leaf_node, v):
    z = leaf_node
    node = leaf_node.parent
    if not node:
        return
    x = 0

    while node:
        x += 1
        action_index = Config.MOVETOINDEX[node.taken_action]
        node.N_update(action_index)
        node.W_update(v, action_index)
        node = node.parent
