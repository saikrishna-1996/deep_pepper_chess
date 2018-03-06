import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#fully connected (including the first layer to hidden layer neurons. so, this is different from giraffe network) network with 2 hidden layers.
#The input is 363-dimensional feature vector as used in the evaluation network for giraffe. The output is 4096-layer. 64x64 = 4096. So, this entire vector indicates the probability of moving from one square to the other square. We know that a piece can't move up from a square and land in the same square. So, technically, it should only be of dimension (4096-64)x1, but, for sanitation purposes let's keep it the way it is.
#ToDo: compute a value function as well using this network
class PolicyNetwork_Full(nn.module):
    def __init__(self, d_in, h1, h2, d_out):
        "We instantiate the linear m odules and assign them as member variables"
        super(PolicyNetwork_Full, self).__init__()
        self.linear1 = nn.Linear(d_in, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, d_out)

    def forward(self, x):
        "Here, we can use modules defined in the constructor (__init__ part defined above) as well as arbitrary operators on Variables"
        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        p_out = F.sigmoid(self.linear3(h2_relu)) #We use sigmoid at the output coz we output probability measures
        return p_out

class PolicyNetwork_Giraffe(nn.module):
    def __init__(self, d_in, gf, pc, sc, h1a, h1b, h2c, h2, d_out):
        "We instantiate various modules"


class PolicyValNetwork_Full(nn.module):
    def __init_(self, d_in, h1, h2p, h2e, d_out, eval_out=1):
        #h2a are the no.of hidden neurons in the second layer to be used for policy network
        #h2b are the no.of hidden neurons in the second layer to be used for evaluation network. evaluation network outputs the value function which is a scalar.
        super(PolicyValNetwork_Full, self).__init__()
        self.linear1 = nn.Linear(d_in, h1)
        self.linear2p = nn.Linear(h1, h2p)
        self.linaer2e = nn.Linear(h1, h2e)
        self.linear3p = nn.Linear(h2p, d_out)
        self.linear3e = nn.Linear(h2e, eval_out)

    def forward(self, x):
        "Here, we will use modules defined in the constructor (__init__ part defined above) as well as arbitray operators on Variables"
        h1_relu = F.relu(self.linear1(x))

        h2p_relu = F.relu(self.linear2p(h1_relu))
        p_out = F.sigmoid(self.linear3p(h2p_relu))

        h2e_relu = F.relu(self.linear2e(h1_relu))
        v_out = self.linear3p(h2e_relu)

        return p_out, v_out



bs = 1  #batch size
d_in = 363
h1 = 1024   #neurons in first hidden layer
h2 = 2048   #nuerons in second hidden layer
h2p = 2048  #nuerons in second hidden layer of policy network
h2e = 512   #neurons in second hidden layer of evaluation network
d_out = 4096 #without including under promotions. otherwise we have to increase

#splitting the giraffe's feature vector to be input to the network
global_features = 17
## the following constitute the global features:
# side to move = 1
# castling rights = 4
# material configuration = 12
piece_centric = 218
## the following constitute the piece-centric features:
# piece lists with their properties = 2*(1+1+2+2+2+8)*5 = 160
# sldiing pieces mobility = 2*(8+4+4+4+4) = 48
# And, I just added extra 10 because, otherwise they are not adding up to 363. Someone pls recheck this.
square_centric = 128
## the following constitute the square-centric features:
# attack map = 64
# defend map = 64
h1a = 32 #no.of first set of neurons in first hidden layer
h2a = 512 #no.of second set of neurons in second hidden layer
h3a = 480 #no.of third set of neurons in third hidden layer

# x is your 363-dimensional input. and y is our output. We are randomly initializing them here.
x = Variable(torch.randn(bs, d_in))
y = Variable(torch.randn(bs, d_out), requires_grad=False)

model = PolicyNetwork_Full(d_in, h1, h2, d_out)
model2 = PolicyNetwork_Giraffe(d_in, global_features, piece_centric, square_centric, h1a, h2a, h3a, h2, d_out)
model3 = PolicyValNetwork_Full(d_in, h1, h2p, h2e, d_out)
