import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        """
        如果 in_channels 不是 g 的整数倍，会触发 ValueError: in_channels must be divisible by groups。
        这里做一个判断，若 in_channels 不能被 g 整除，则将 g 自动设为1。
        或者可以根据需要手动调整 g，保证 in_channels%g=0。
        """
        super(DualConv, self).__init__()

        if in_channels % g != 0:
            print(f"[WARNING] in_channels={in_channels} 不可被 groups={g} 整除，已强制将 g=1")
            g = 1

        # Group Convolution
        self.gc = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            groups=g, 
            bias=False
        )
        self.gc_bn = nn.BatchNorm2d(out_channels)

        # Pointwise Convolution (1×1)
        self.pwc = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            bias=False
        )
        self.pwc_bn = nn.BatchNorm2d(out_channels)

        # 也可以考虑将激活函数改为其他形式，如 nn.SiLU()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_data):
        """
        将分组卷积与1×1卷积的结果相加。
        """
        gc_out = self.gc_bn(self.gc(input_data))
        pwc_out = self.pwc_bn(self.pwc(input_data))
        out = gc_out + pwc_out
        out = self.activation(out)
        return out
 
########################################DualConv end ########################################