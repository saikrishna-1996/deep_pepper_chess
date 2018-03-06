import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#This is not in the main scope of the project, but feel free to train it using TD-lambda or whatever. But, pretraining with stockfish is highly recommended. look at python-chess documents for more info.
#fully connected (including the first layer to hidden layer neurons. so, this is different from giraffe network) network with 2 hidden layers.
#The input is 363-dimensional feature vector as used in the evaluation network for giraffe. The output is a single scalar value
class ValueNetwork_Full(nn.module):
    def __init__(self, d_in, h1, h2, d_out):
        "We instantiate the linear m odules and assign them as member variables"
        super(ValueNetwork_Full, self).__init__()
        self.linear1 = nn.Linear(d_in, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, d_out)

    def forward(self, x):
        "Here, we can use modules defined in the constructor (__init__ part defined above) as well as arbitrary operators on Variables"
        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu)) #But, if we are using ReLU, and if you evaluation from black's turn, it might not be easy for the network. So, we have to invert the board at every turn?!
        y_out = self.linear3(h2_relu)
        return y_out


bs = 1  #batch size
d_in = 363
h1 = 1024   #neurons in first hidden layer
h2 = 2048   #nuerons in second hidden layer
d_out = 1 #without including under promotions. otherwise we have to increase

# x is your 363-dimensional input. and y is our output. We are randomly initializing them here.
x = Variable(torch.randn(bs, d_in))
y = Variable(torch.randn(bs, d_out), requires_grad=False)

model = ValueNetwork_Full(d_in, h1, h2, d_out)

