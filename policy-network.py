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
        y_out = F.sigmoid(self.linear3(h2_relu)) #We use sigmoid at the output coz we output probability measures
        return y_out

class PolicyNetwork_Giraffe(nn.module):


bs = 1  #batch size
d_in = 363
h1 = 1024   #neurons in first hidden layer
h2 = 2048   #nuerons in second hidden layer
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
sqaure_centric = 128
## the following constitute the square-centric features:
# attack map = 64
# defend map = 64

# x is your 363-dimensional input. and y is our output. We are randomly initializing them here.
x = Variable(torch.randn(bs, d_in))
y = Variable(torch.randn(bs, d_out), requires_grad=False)

model = PolicyNetwork_Full(d_in, h1, h2, d_out)
model2 = PolicyNetwork_Giraffe(d_in, global_features, piece_centric, square_centric,
