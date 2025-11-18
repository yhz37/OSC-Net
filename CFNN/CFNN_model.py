""" Full assembly of the parts to form the complete network """

from .CFNN_parts import *
import torch

class CFNN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,Num_F=None):
        super(CFNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        num_ch0 = 64
        num_ch1 = 128
        num_ch2 = 32
        num_ch3 = 8
        self.inc = DoubleConv(n_channels, num_ch0)
        self.down1 = Down(num_ch0, num_ch1)
        self.down2 = Down(num_ch1, num_ch2)
        self.down3 = Down(num_ch2, num_ch3)
        # self.down4 = Down(32, 32)
        # self.down5 = Down(32, 32)
        # self.down6 = Down(32, 16)

        self.fc1 = nn.Sequential(nn.Linear(int(Num_F/2**3*num_ch3)+1, 512),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 128),nn.ReLU()) 
        self.fc3 = nn.Sequential(nn.Linear(128, n_classes)) 

    def forward(self, x,path='all'):
        if path=='all':
            xs,xr = torch.split(x, [x.shape[2]-1,1],2)
            x1 = self.inc(xs)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            # x4 = self.down4(x4)
            # x4 = self.down5(x4)
            # x4 = self.down6(x4)
            x5 = torch.flatten(x4, 1)
            xr = torch.flatten(xr[:,0,:], 1)
            x6 = self.fc1(torch.cat((x5, xr),1))
            x7 = self.fc2(x6)
            y = self.fc3(x7)
            return y
class CFNN_hete(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,Num_F=None):
        super(CFNN_hete, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        num_ch0 = 64
        num_ch1 = 128
        num_ch2 = 32
        num_ch3 = 8
        self.inc = DoubleConv(n_channels, num_ch0)
        self.down1 = Down(num_ch0, num_ch1)
        self.down2 = Down(num_ch1, num_ch2)
        self.down3 = Down(num_ch2, num_ch3)
        # self.down4 = Down(32, 32)
        # self.down5 = Down(32, 32)
        # self.down6 = Down(32, 16)

        self.fc1 = nn.Sequential(nn.Linear(int(Num_F/2**3*num_ch3)+1, 512),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 128),nn.ReLU()) 
        self.fc3 = nn.Sequential(nn.Linear(128, n_classes)) 

    def forward(self, x,path='all'):
        if path=='all':
            xs,xr = torch.split(x, [x.shape[2]-1,1],2)
            x1 = self.inc(xs)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            # x4 = self.down4(x4)
            # x4 = self.down5(x4)
            # x4 = self.down6(x4)
            x5 = torch.flatten(x4, 1)
            xr = torch.flatten(xr[:,0,:], 1)
            x6 = self.fc1(torch.cat((x5, xr),1))
            x7 = self.fc2(x6)
            y = self.fc3(x7)
            Num_Split = int(y.size(dim=1)/2)
            logits,s = torch.split(y,[Num_Split,Num_Split], 1)
            return logits,s