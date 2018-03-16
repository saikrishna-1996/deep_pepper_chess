import chess
import torch
from torch.autograd import Variable
import policy_network
import features
import mcts
from policy_network import PolicyValNetwork_Giraffe as pvng
import config
import os
import glob


def do_backprop(featuress, model):

    #first convert this batch_board to batch_features
    #batch_board should be of dimension (batch_size, board)
    #batch_feature = Variable(torch.randn(batch_size, 353))
    criterion1 = torch.nn.MSELoss(size_average = False)
    criterion2 = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4, momentum=0.9)

    ##We can do this cuda part later!?
    #if torch.cuda_is_available():
    #    criterion.cuda()
    #    policy_network.PolicyValNetwork_Giraffe = policy_network.PolicyValNetwork.cuda()

    #for i in range(batch_size):
    #    batch_feature[i,:] = features.BoardToFeature(batch_board[i,board])

    #pvng_model = pvng(d_in, gf, pc, sc, h1a, h1b, h1c, h2p, h2e, d_out, eval_out=1)
    nn_policy_out, nn_val_out = pvng_model(features)
    mcts_policy_out = ahmad_give_me_this()
    mcts_val_out = ahmad_give_me_this_too()

    loss1 = criterion1(mcts_val_out, nn_val_out)
    loss2 = criterion2(mcts_policy_out, nn_policy_out)

    l2_reg = None
    for wei in pvng_model.parameters():
        if l2_reg is None:
            l2_reg = wei.norm(2)
        else:
            l2_reg = l2_reg + wei.norm(2)
    loss3 = 0.1*l2_reg

    loss = loss1 + loss2 + loss3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save(model,fname,network_iter):
    save_info = {
        'state_dict': model.state_dict(),
        'iteration': network_iter
    }
    fpath = os.path.join(config.NETPATH,fname)
    torch.save(save_info,fname)


def load(best=False):
    if (best):
        best_fname = config.BESTNET_NAME
        model_state = torch.load(config.NETPATH,best_fname)
    else:
        #get newest file
        list_of_files = glob.glob(config.NETPATH)
        latest_file = max(list_of_files, key=os.path.getctime)
        model_state = torch.load(latest_file)
    return model_state
    