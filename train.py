import glob
import os

import numpy as np
import torch

# There will need to be some function that calls both of these functions and uses the output from load_gamefile to train a network
# load_gamefile will return a list of lists containing [state, policy, value] as created in MCTS.
from config import Config
from policy_network import PolicyValNetwork_Giraffe


def load_gamefile(net_number):  # I'm not married to this, I think it could be done better.
    list_of_files = glob.glob(Config.GAMEPATH)
    net_files = []
    for file_name in list_of_files:
        if 'p' + repr(net_number) in file_name:
            net_files.append(file_name)

    index = np.random.randint(0, len(net_files))
    try:
        return np.load(net_files[index])
    except IOError:
        print('Could not load gamefile!')


def train_model(model=PolicyValNetwork_Giraffe(), games=None, net_number=0, min_num_games=400):
    if games is None:
        game_data = load_gamefile(net_number)
    else:
        game_data = games

    for games_trained in range(min_num_games):
        if game_data is not None:
            for data in game_data[0]:
                features = torch.from_numpy(np.array(data[0]))
                do_backprop(features, data[1], data[2], model)


def do_backprop(features, policy, act_val, model):
    # first convert this batch_board to batch_features
    # batch_board should be of dimension (batch_size, board)
    # batch_feature = Variable(torch.randn(batch_size, 353))
    criterion1 = torch.nn.MSELoss(size_average=False)
    criterion2 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ##We can do this cuda part later!?
    # if torch.cuda_is_available():
    #    criterion.cuda()
    #    policy_network.PolicyValNetwork_Giraffe = policy_network.PolicyValNetwork.cuda()

    # for i in range(batch_size):
    #    batch_feature[i,:] = features.BoardToFeature(batch_board[i,board])

    # pvng_model = pvng(d_in, gf, pc, sc, h1a, h1b, h1c, h2p, h2e, d_out, eval_out=1)
    features = features.view(1,-1)
    #act_val = torch.autograd.Variable(act_val)
    #policy = torch.autograd.Variable(policy)
    nn_policy_out, nn_val_out = model(features)

    loss1 = criterion1( nn_val_out,act_val)
    loss2 = criterion2(nn_policy_out,policy)

    l2_reg = None
    for weight in model.parameters():
        if l2_reg is None:
            l2_reg = weight.norm(2)
        else:
            l2_reg = l2_reg + weight.norm(2)
    loss3 = 0.1 * l2_reg

    loss = loss1 + loss2 + loss3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save(model, fname, network_iter):
    save_info = {
        'state_dict': model.state_dict(),
        'iteration': network_iter
    }
    fpath = os.path.join(Config.NETPATH, fname)
    torch.save(save_info, fpath)


def load(best=False):
    if best:
        best_fname = Config.BESTNET_NAME
        try:
            model_state = torch.load(Config.NETPATH, best_fname)
        except:
            print("Could not load model")
            return None
    else:
        # get newest file
        list_of_files = glob.glob(Config.NETPATH)
        latest_file = max(list_of_files, key=os.path.getctime)
        try:
            model_state = torch.load(latest_file)
        except:
            print("Could not load file.")
            return None
    return model_state
