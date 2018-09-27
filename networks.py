import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import Config

# fully connected (including the first layer to hidden layer neurons. so, this is different from giraffe network) network with 2 hidden layers.
# The input is 352-dimensional feature vector. The output is 5120-layer. 64x64 = 4096 + additional outputs to accomodate exception moves likes pawn promotions. So, this entire vector indicates the probability of moving from one square to the other square.


class PolicyNetwork_Full(nn.Module):
    def __init__(self,
            pretrain = False,
            d_in = Config.d_in,
            h1 = Config.h1,
            h2 = Config.h2,
            d_out = Config.d_out):

        super(PolicyNetwork_Full, self).__init__()

        self.linear1 = nn.Linear(d_in, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, d_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):

        x = Variable(x)

        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        p_out = F.sigmoid(self.linear3(h2_relu))  # We use sigmoid at the output coz we output probability measures

        return p_out


class PolicyNetwork_Giraffe(nn.Module):
    def __init__(self,
            pretrain = False,
            d_in = Config.d_in,
            gf = Config.global_features,
            pc = Config.piece_centric,
            sc = Config.square_centric,
            h1a = Config.h1a,
            h1b = Config.h1b,
            h1c = Config.h1c,
            h2 = Config.h2,
            d_out = Config.d_out):

        super(PolicyNetwork_Giraffe, self).__init__()

        self.gf = gf
        self.pc = pc
        self.sc = sc
        self.linear1a = nn.Linear(gf, h1a)
        self.linear1b = nn.Linear(pc, h1b)
        self.linear1c = nn.Linear(sc, h1c)
        self.linear2 = nn.Linear(h1a + h1b + h1c, h2)
        self.linear3 = nn.Linear(h2, d_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):

        x = Variable(x)

        gf = self.gf
        pc = self.pc
        sc = self.sc

        x1 = x[:, 0:gf - 1]
        x2 = x[:, gf:gf + pc - 1]
        x3 = x[:, gf + pc:gf + pc + sc - 1]

        h1a_relu = F.relu(self.linear1a(x1))
        h1b_relu = F.relu(self.linear1b(x2))
        h1c_relu = F.relu(self.linear1c(x3))
        h1_relu = torch.cat(h1a_relu, h1b_relu, h1c_relu, dim=1)
        h2_relu = F.relu(self.linear2(h1_relu))
        p_out = F.sigmoid(self.linear3(h2_relu))

        return p_out


class PolicyValNetwork_Giraffe(nn.Module):
    def __init__(self,
                 pretrain=False,
                 d_in=Config.d_in,
                 gf=Config.global_features,
                 pc=Config.piece_centric,
                 sc=Config.square_centric,
                 h1a=Config.h1a,
                 h1b=Config.h1b,
                 h1c=Config.h1c,
                 h2p=Config.h2p,
                 h2e=Config.h2e,
                 d_out=Config.d_out,
                 eval_out=1):

        super(PolicyValNetwork_Giraffe, self).__init__()

        self.gf = gf
        self.pc = pc
        self.sc = sc

        self.linear1a = nn.Linear(gf, h1a)
        self.linear1b = nn.Linear(pc, h1b)
        self.linear1c = nn.Linear(sc, h1c)
        self.linear2p = nn.Linear(h1a + h1b + h1c, h2p)
        self.linear2e = nn.Linear(h1a + h1b + h1c, h2e)
        self.linear3p = nn.Linear(h2p, d_out)
        self.linear3e = nn.Linear(h2e, eval_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):

        x = Variable(x.float())

        gf = self.gf
        pc = self.pc
        sc = self.sc

        x1 = x[:, 0:gf]
        x2 = x[:, gf:gf + pc]
        x3 = x[:, gf + pc:gf + pc + sc]

        h1a_relu = F.relu(self.linear1a(x1))
        h1b_relu = F.relu(self.linear1b(x2))
        h1c_relu = F.relu(self.linear1c(x3))
        h1_relu = torch.cat((h1a_relu, h1b_relu, h1c_relu), dim=1)

        h2p_relu = F.relu(self.linear2p(h1_relu))
        p_out = F.log_softmax(self.linear3p(h2p_relu), dim=1)

        h2e_relu = F.relu(self.linear2e(h1_relu))
        val_out = Config.GAME_SCORE*F.tanh(self.linear3e(h2e_relu))

        return p_out, val_out


class PolicyValNetwork_Full(nn.Module):
    def __init__(self,
            pretrain = False,
            d_in = Config.d_in,
            h1 = Config.h1,
            h2p = Config.h2p,
            h2e = Config.h2e,
            d_out = Config.d_out,
            eval_out=1):

        # h2a are the no.of hidden neurons in the second layer to be used for policy network
        # h2b are the no.of hidden neurons in the second layer to be used for evaluation network. evaluation network outputs the value function which is a scalar.
        super(PolicyValNetwork_Full, self).__init__()

        self.linear1 = nn.Linear(d_in, h1)
        self.linear2p = nn.Linear(h1, h2p)
        self.linaer2e = nn.Linear(h1, h2e)
        self.linear3p = nn.Linear(h2p, d_out)
        self.linear3e = nn.Linear(h2e, eval_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):

        h1_relu = F.relu(self.linear1(x))

        h2p_relu = F.relu(self.linear2p(h1_relu))
        p_out = F.sigmoid(self.linear3p(h2p_relu))

        h2e_relu = F.relu(self.linear2e(h1_relu))
        v_out = F.tanh(self.linear3p(h2e_relu))

        return p_out, v_out

## Start defining value networks

class Critic_Giraffe(nn.Module):
    def __init__(self,
            pretrain = False,
            d_in = Config.d_in,
            gf = Config.global_features,
            pc = Config.piece_centric,
            sc = Config.square_centric,
            h1a = Config.h1a,
            h1b = Config.h1b,
            h1c = Config.h1c,
            h2 = Config.h2,
            eval_out=1):

        super(Critic_Giraffe, self).__init__()

        self.gf = gf
        self.pc = pc
        self.sc = sc
        self.linear1a = nn.Linear(gf, h1a)
        self.linear1b = nn.Linear(pc, h1b)
        self.linear1c = nn.Linear(sc, h1c)
        self.linear2 = nn.Linear(h1a + h1b + h1c, h2)
        self.linear3 = nn.Linear(h2, eval_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):

        x = Variable(x.float())

        gf = self.gf
        pc = self.pc
        sc = self.sc

        x1 = x[:, 0:gf]
        x2 = x[:, gf:gf + pc]
        x3 = x[:, gf + pc:gf + pc + sc]

        h1a_relu = F.relu(self.linear1a(x1))
        h1b_relu = F.relu(self.linear1b(x2))
        h1c_relu = F.relu(self.linear1c(x3))
        h1_relu = torch.cat(h1a_relu, h1b_relu, h1c_relu, dim=1)
        h2_relu = F.relu(self.linear2e(h1_relu))
        val_out = F.Tanh(self.linear3e(h2_relu))

        return val_out


class Critic_FCGiraffe(nn.Module): # Critic fully connected
    def __init_(self,
            pretrain = False,
            d_in = Config.d_in,
            h1 = Config.h1,
            h2 = Config.h2,
            eval_out=1):

        super(Critic_FCGiraffe, self).__init__()

        self.linear1 = nn.Linear(d_in, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, eval_out)

        if pretrain:
            pretrain(self)

    def forward(self, x):
        x = Variable(x.float())

        h1_relu = F.relu(self.linear1(x))
        h2_relu = F.relu(self.linear2(h1_relu))
        v_out = F.Tanh(self.linear3(h2_relu))

        return v_out