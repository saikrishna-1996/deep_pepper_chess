
from train.MCTS import Node

def human_play(root, explore_factor):

    root.env.render()
    legal_move = [move.uci() for move in root.env.legal_moves]
    action = None
    while not(action in legal_move):
        action = input('Enter your move:\n')


    next_env = root.env.copy()
    next_env.step(action)
    # if not root.children:
    #     return Node(next_env, explore_factor), root
    try:
        children_boards = [env.board for env in root.children]
        if  next_env.board in children_boards:
            return root.children[children_boards.index(next_env.board)], root
    except:
        return Node(next_env, explore_factor), root

