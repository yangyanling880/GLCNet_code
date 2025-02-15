import torch
import torch.nn as nn
import sys
current_module = sys.modules[__name__]
import numpy as np
from scipy.signal import hilbert
current_module = sys.modules[__name__]
import sys
from torch_geometric.nn import GCNConv
from networks3 import *


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SquareLayer(nn.Module):
    '''
    The Square layer: claculates the Square of the data.
    '''

    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x * x


class LogLayer(nn.Module):
    '''
    The log layer: claculates the log of the data.
    '''

    def __init__(self):
        super(LogLayer, self).__init__()

    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

class PLILayer(nn.Module):
    def __init__(self):
        super(PLILayer, self).__init__()

    @staticmethod
    def compute_pli(x, y):
        thetax = torch.angle(torch.tensor(hilbert(x.numpy())))
        thetay = torch.angle(torch.tensor(hilbert(y.numpy())))
        thetadiff = thetax - thetay
        thetadiffmodify = torch.angle(torch.exp(1j * thetadiff))
        thetadiffmodify[(thetadiffmodify == np.pi) | (thetadiffmodify == -np.pi)] = 0
        return torch.abs(torch.mean(torch.sign(thetadiffmodify)))

    def forward(self, x):
        batch_size, channels, samples = x.shape
        PLI = torch.zeros((batch_size, channels, channels), dtype=torch.float32)
        for batch in range(batch_size):
            for i in range(channels):
                for j in range(channels):
                    if i != j:
                        PLI[batch, i, j] = self.compute_pli(x[batch, i, :], x[batch, j, :])
        return PLI



class GLCNet(nn.Module):


    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):

        return nn.Sequential(
            Conv2dWithConstraint(nBands, m * nBands, (nChan, 1), groups=nBands,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish()
        )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))



    def __init__(self, nChan, nTime, nClass=2, nBands=9, m=32,
                 temporalLayer='LogVarLayer', strideFactor=5, doWeightNorm=True, *args, **kwargs):
        super(GLCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm=doWeightNorm)

        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m * self.nBands * self.strideFactor, nClass, doWeightNorm=doWeightNorm)
        self.GCNLayer=GcnNet(9)
        self.TimeLSTMLayer = nn.LSTM(400,5)


    def forward(self, x,adj):

        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x2=self.GCNLayer(x,torch.tensor(adj,device='cuda:0'))
        x = self.scb(x)
        x3 = torch.unsqueeze(self.TimeLSTMLayer(torch.squeeze(x,dim=2))[0],dim=-1)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)+x2.transpose(-1,-2)  + x3
        x = torch.flatten(x, start_dim=1)
        x = self.lastLayer(x)

        return x



