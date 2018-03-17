import numpy as np
import os
#this is hypothetical functions and classes that should be created by teamates.
import chess.uci
from policy_network import PolicyValNetwork_Full
import value_network
from chess_env import ChessEnv
from heuristics import stockfish_eval
from features import BoardToFeature
import config
from MCTS import MCTS
import train

def Generating_games():
    triplet=[]
    model = PolicyValNetwork_Full(config.d_in, config.h1, config.h2p, config.h2e, config.d_out)
    old_net_iter = 0
    game_number = 0
    while (True):
        net_stats = train.load(True)
        if (net_stats != None):
            net_iter = net_stats['iteration']
            if ((net_iter != old_net_iter) and (game_number > config.MINGAMES)):
                game_number = 0
                model = model.load_state_dict(net_stats['state_dict'])
                old_net_iter = net_iter
        else:
            net_iter = 0
        step_game = 0
        temperature = 1
        env = ChessEnv()
        env.reset()
        while not env.game_over()[0]:
            state = env.board
            step_game +=1
            if step_game == config.TEMP_REDUCE_STEP:
                temperature = 10e-6
            pi = MCTS(env,
                      init_W=np.zeros((4096,)),
                      init_N=np.zeros((4096,)),
                      explore_factor = config.EXPLORE_FACTOR,
                      temp=temperature,
                      network=model,
                      dirichlet_alpha=config.D_ALPHA,
                      epsilon = config.EPS)

            action_index = np.argmax(pi)
            triplet.append([state,pi])

            env.step( config.INDEXTOMOVE[action_index])

        z = env.game_over()[1] #from white perspective

        for i in range(len(triplet)-step_game, len(triplet)):
            triplet[i].append( z )
        np.save(os.path.join(config.GAMEPATH,'p'+ net_iter + '_g' + str(game_number)), np.array(triplet))
        triplet = []
        game_number += 1

