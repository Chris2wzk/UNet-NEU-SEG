import torch
from torch import nn
import torch.nn.functional as F

######################################## Common ########################################
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

######################################## Dynamic Group Convolution Shuffle Transformer ########################################
class DGCST(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
 
        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)
 
        x1 = self.gconv(x1)
 
        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)
 
        x = torch.cat([y, x2], 1)
        return x + self.conv2(x)
 
######################################## Dynamic Group Convolution Shuffle Transformer end ########################################