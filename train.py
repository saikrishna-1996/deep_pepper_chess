import chess
import torch
from torch.autograd import Variable
import policy_network
import features
import MCTS
from policy_network import PolicyValNetwork_Giraffe as pvng
import config
import os
import glob
import np

#There will need to be some function that calls both of these functions and uses the output from load_gamefile to train a network
#load_gamefile will return a list of lists containing [state, policy, value] as created in MCTS.
def load_gamefile(net_number): #I'm not married to this, I think it could be done better.
    list_of_files = glob.glob(config.GAMEPATH)
    net_files = []
    for file_name in list_of_files:
        if 'p'+repr(net_number) in file_name:
            net_files.append(file_name)
    
    index = np.random.randint(0,len(net_files))
    try:
        triplet = np.load(net_files[index])
    except:
        print('Could not load gamefile!')
    return triplet
        

def train_model(model, net_number, min_num_games):
    games_trained = 0
    while (games_trained < min_num_games):
        game_data = load_gamefile(net_number)
        if game_data != None:
            for state, policy, val in game_data:
                do_backprop(state,val,policy,model)


def do_backprop(features,policy,act_val, model):

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
        try:
            model_state = torch.load(config.NETPATH,best_fname)
        except:
            print("Could not load model")
            return None
    else:
        #get newest file
        list_of_files = glob.glob(config.NETPATH)
        latest_file = max(list_of_files, key=os.path.getctime)
        try:
            model_state = torch.load(latest_file)
        except:
            print("Could not load file.")
            return None
    return model_state

