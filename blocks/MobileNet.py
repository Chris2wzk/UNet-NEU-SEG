import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            conv_bn(n_channels, 32, 1),  # 416,416,3 -> 416,416,32
            conv_dw(32, 64, 1),  # 416,416,32 -> 416,416,64
            conv_dw(64, 128, 2),  # 416,416,64 -> 208,208,128
            conv_dw(128, 128, 1),  # 208,208,128
            conv_dw(128, 256, 2),  # 208,208,128 -> 104,104,256
            conv_dw(256, 256, 1),  # 104,104,256
        )
        self.layer2 = nn.Sequential(
            conv_dw(256, 512, 2),  # 104,104,256 -> 52,52,512
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),  # 52,52,512
        )
        self.layer3 = nn.Sequential(
            conv_dw(512, 1024, 2),  # 52,52,512 -> 26,26,1024
            conv_dw(1024, 1024, 1),  # 26,26,1024
        )

    def forward(self, x):
        # 返回三个特征图，用于UNet的skip connections
        x2 = self.layer1(x)      # 104,104,256
        x1 = self.layer2(x2)     # 52,52,512
        x0 = self.layer3(x1)     # 26,26,1024
        
        return x2, x1, x0        # 从浅层到深层的特征图
