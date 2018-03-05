import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#fully connected (including the first layer to hidden layer neurons. so, this is different from giraffe network) network with 2 hidden layers.
#The input is 363-dimensional feature vector as used in the evaluation network for giraffe. The output is 4096-layer. 64x64 = 4096. So, this entire vector indicates the probability of moving from one square to the other square. We know that a piece can't move up from a square and land in the same square. So, technically, it should only be of dimension (4096-64)x1, but, for sanitation purposes let's keep it the way it is.
#ToDo: compute a value function as well using this network
class PolicyNetwork_Full(nn.module):
    def __init__(self, h1, h2):
        "We instantiate the linear m odules and assign them as member variables"
        super(PolicyNetwork_Full, self).__init__()
        self.linear1 = nn.Linear(363, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, 4096)

    def forward(self, x):
        "Here, we can use modules defined in the constructor (__init__ part defined above) as well as arbitrary operators on Variables"
        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        y_out = F.sigmoid(self.linear3(h2_relu)) #We use sigmoid at the output coz we output probability measures
