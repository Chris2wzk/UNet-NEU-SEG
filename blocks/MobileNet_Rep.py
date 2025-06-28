import torch
import torch.nn as nn
import torch.nn.functional as F
from .FastKan import FastKANConv2DLayer

def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_dw_rep(inp, oup, stride=1):
    return nn.Sequential(
        # 深度可分离卷积
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        
        # 1x1卷积
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class MobileNet_Rep(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet_Rep, self).__init__()
        # 第一层
        self.layer1 = nn.Sequential(
            conv_bn(n_channels, 32, 1),  # 416,416,3 -> 416,416,32
            conv_dw_rep(32, 64, 1),      # 416,416,32 -> 416,416,64
            conv_dw_rep(64, 128, 2),     # 416,416,64 -> 208,208,128
            conv_dw_rep(128, 128, 1),    # 208,208,128
            conv_dw_rep(128, 256, 2),    # 208,208,128 -> 104,104,256
            conv_dw_rep(256, 256, 1),    # 104,104,256
        )
        
        # 简化的FastKAN注意力
        self.attention1 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=1, dropout=0.1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # 第二层
        self.layer2 = nn.Sequential(
            conv_dw_rep(256, 512, 2),    # 104,104,256 -> 52,52,512
            conv_dw_rep(512, 512, 1),
            conv_dw_rep(512, 512, 1),
            conv_dw_rep(512, 512, 1),    # 减少一层
            conv_dw_rep(512, 512, 1),    # 减少一层
        )
        
        # 简化的FastKAN注意力
        self.attention2 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=1, dropout=0.1)
        self.bn2 = nn.BatchNorm2d(512)
        
        # 第三层
        self.layer3 = nn.Sequential(
            conv_dw_rep(512, 1024, 2),   # 52,52,512 -> 26,26,1024
            conv_dw_rep(1024, 1024, 1),  # 26,26,1024
        )
        
        # 简化的FastKAN注意力
        self.attention3 = FastKANConv2DLayer(1024, 1024, kernel_size=3, padding=1, dropout=0.1)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        # 返回三个特征图，每个特征图都经过FastKAN注意力增强
        x2 = self.layer1(x)          # 104,104,256
        x2_att = self.attention1(x2)
        x2 = self.bn1(x2_att + x2)
        
        x1 = self.layer2(x2)         # 52,52,512
        x1_att = self.attention2(x1)
        x1 = self.bn2(x1_att + x1)
        
        x0 = self.layer3(x1)         # 26,26,1024
        x0_att = self.attention3(x0)
        x0 = self.bn3(x0_att + x0)
        
        return x2, x1, x0