""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

# 一行就能把 blocks 文件夹下所有 py 中的符号一次性导入
from blocks import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


######################### NAMAttention加到卷积层 开始 ###############################


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU()
        #self.act = nn.LeakyReLU(0.1)
        self.act = nn.ReLU(inplace=True)
        #self.act = MetaAconC(c2)
        #self.act = AconC(c2)
        #self.act = Mish()
        #self.act = Hardswish()
        #self.act = FReLU(c2)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def fuseforward(self, x):
        return self.act(self.conv(x))
 
class DoubleConvAttentionNAM(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = NAMAttention(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionNAM(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionNAM(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### NAMAttention加到卷积层 结束 ###############################

######################### GAMAttention加到卷积层 开始 ###############################

 
class DoubleConvAttentionGAM(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = GAMAttention(mid_channels,mid_channels)
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionGAM(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionGAM(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### GAMAttention加到卷积层 结束 ###############################

######################### EMA加到卷积层 开始 ###############################

class DoubleConvAttentionEMA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = EMA(mid_channels)

 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionEMA(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionEMA(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### EMA加到卷积层 结束 ###############################

######################### SimAM加到卷积层 开始 ###############################

class DoubleConvAttentionSimAM(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = SimAM()
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionSimAM(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionSimAM(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### SimAM加到卷积层 结束 ###############################

######################### SpatialGroupEnhance加到卷积层 开始 ################

class DoubleConvAttentionSGE(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)

        self.attention = SpatialGroupEnhance()
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionSGE(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionSGE(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### SpatialGroupEnhance加到卷积层 结束 ################

######################### LSKBlock加到卷积层 开始 ################

class DoubleConvAttentionLSKBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)

        self.attention = LSKBlock(mid_channels)
 
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionLSKBlock(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionLSKBlock(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### LSKBlock加到卷积层 结束 ################

######################### CPCA加到卷积层 开始 ###################

 
class DoubleConvAttentionCPCA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = CPCA(mid_channels)
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionCPCA(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionCPCA(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### CPCA加到卷积层 结束 ###################

######################### MLCA加到卷积层 开始 ###################

class DoubleConvAttentionMLCA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = MLCA(mid_channels)
 
 
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionMLCA(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionMLCA(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### MLCA加到卷积层 结束 ################### 

######################### RepVGG加到卷积层 开始 ###################
    
class DoubleConvAttentionRepVGG(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = RepVGG(mid_channels,mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionRepVGG(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionRepVGG(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### RepVGG加到卷积层 结束 ###################

######################### SEAttention加到卷积层 开始 ################### 
 
class DoubleConvAttentionSE(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        #self.conv1 = PConv(in_channels, mid_channels)
        self.attention = SEAttention(mid_channels)
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionSE(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionSE(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### SEAttention加到卷积层 结束 ###################

######################### TripletAttention加到卷积层 开始 ###################


class DoubleConvAttentionTriplet(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = TripletAttention()
 
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
 
class DownAttentionTriplet(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionTriplet(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### TripletAttention加到卷积层 结束 ###################

######################### shuffleAttention加到卷积层 开始 ###################

class DoubleConvAttentionShuffle(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = ShuffleAttention(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x

class DownAttentionShuffle(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionShuffle(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### shuffleAttention加到卷积层 结束 ###################

######################### RepLKBlock加到卷积层 开始 ##################

 
class DoubleConvAttentionRepLK(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = RepLKBlock(mid_channels,mid_channels,31, 5, 0.0, False)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionRepLK(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionRepLK(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### RepLKBlock加到卷积层 结束 ##################

######################### DiverseBranchBlock加到卷积层 开始 ##############################

class DoubleConvAttentionDBB(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        #self.attention = CBAM(mid_channels)
        #self.attention = NAMAttention(mid_channels)
        self.attention = DiverseBranchBlock(mid_channels, mid_channels, 1, 1)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x

class DownAttentionDBB(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionDBB(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### DiverseBranchBlock加到卷积层 结束 ##############################

######################### iRMB加到卷积层 开始 ############################## 
class DoubleConvAttentioniRMB(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = iRMB(mid_channels, mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentioniRMB(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentioniRMB(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### iRMB加到卷积层 结束 ##############################

######################### DGCST加到卷积层 开始 ##############################

 
class DoubleConvAttentionDGCST(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        #self.attention = CBAM(mid_channels)
        #self.attention = NAMAttention(mid_channels)
        self.attention = DGCST(mid_channels, mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionDGCST(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionDGCST(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### DGCST加到卷积层 结束 ##############################

######################### PPA加到卷积层 开始 ##############################
 
class DoubleConvAttentionPPA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = PPA(mid_channels, mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x

class DownAttentionPPA(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionPPA(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

######################### PPA加到卷积层 结束 ##############################

######################### UIBB加到卷积层 开始 ##############################
 
class DoubleConvAttentionUIBB(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = UniversalInvertedBottleneckBlock(mid_channels, mid_channels,0, 3, True, 1, 2)
 
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionUIBB(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionUIBB(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### UIBB加到卷积层 结束 ##############################

######################### CAA加到卷积层 开始 ############################## 
class DoubleConvAttentionCAA(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = CAA(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionCAA(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionCAA(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### CAA加到卷积层 结束 ##############################

######################### StarBlock加到卷积层 开始 ########################
    
class DoubleConvAttentionStarBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = Star_Block(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionStarBlock(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionStarBlock(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### StarBlock加到卷积层 结束 ########################

#########################SSPCAB加到卷积层 开始 ##############################
 
class DoubleConvAttentionSSPCAB(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        #self.attention = CBAM(mid_channels)
        self.attention = SSPCAB(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionSSPCAB(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionSSPCAB(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### SSPCAB加到卷积层 结束 ##############################

######################### CGLU加到卷积层 开始 ##############################
 
class DoubleConvAttentionCGLU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        #self.attention = CBAM(mid_channels)
        self.attention = ConvolutionalGLU(mid_channels)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x
class DownAttentionCGLU(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvAttentionCGLU(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### CGLU加到卷积层 结束 ##############################




####################################################################
######################### PConv加到DoubleConv 开始 ##################
class DoubleConv_PConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2 的基本思路（不修改PConv本身的源码）：
    1) 如果 in_channels != mid_channels，就先用普通卷积把通道 in_channels => mid_channels
    2) 调用 PConv(dim=mid_channels, n_div=...)，它只能维持同样的dim，不做升降维
    3) 如果 mid_channels != out_channels，就再用普通卷积把通道 mid_channels => out_channels

    注意：PConv 的写法是 (dim, n_div, ...)，若 mid_channels < n_div，会出现 0 通道的错误。
         因此建议确保 mid_channels >= n_div。示例：若 mid_channels=64, n_div=2，则部分卷积时
         dim_conv=32, dim_untouched=32；不会报错。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, n_div=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # 若 mid_channels < n_div，则会导致 PConv 里出现 0 通道报错，这里可加一条断言
        if mid_channels < n_div:
            raise ValueError(
                f"mid_channels({mid_channels}) must be >= n_div({n_div}) to avoid 0 channels in PConv."
            )

        # 第一步：普通卷积（可升/降维） in_channels => mid_channels
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 第二步：调用原始 PConv 不修改其源码。dim=mid_channels，意味着不做升降维，只拆分通道做卷积
        self.pconv = PConv(
            dim=mid_channels,
            n_div=n_div,
            forward="split_cat",
            kernel_size=3
        )
        self.pconv_bn = nn.BatchNorm2d(mid_channels)
        self.pconv_act = nn.ReLU(inplace=True)

        # 第三步：如需把通道进一步变成 out_channels，则再加一个普通卷积
        self.post_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # pre_conv: in_channels => mid_channels
        x = self.pre_conv(x)
        # pconv: mid_channels => mid_channels（仅拆分部分通道进行卷积）
        x = self.pconv_act(self.pconv_bn(self.pconv(x)))
        # post_conv: mid_channels => out_channels（如果 mid_channels==out_channels，可省略）
        x = self.post_conv(x)
        return x

class Down_PConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_PConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
######################### PConv加到DoubleConv 结束 ##################
class DoubleConv_PConv_RepLK(nn.Module):
    """
    将 PConv 和 RepLK 注意力机制结合：
    1) 保持 PConv 的三阶段结构
    2) 在 PConv 后添加 RepLK 注意力
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, n_div=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # 检查通道数约束
        if mid_channels < n_div:
            raise ValueError(
                f"mid_channels({mid_channels}) must be >= n_div({n_div}) to avoid 0 channels in PConv."
            )

        # 第一步：预处理卷积
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 第二步：PConv
        self.pconv = PConv(
            dim=mid_channels,
            n_div=n_div,
            forward="split_cat",
            kernel_size=3
        )
        self.pconv_bn = nn.BatchNorm2d(mid_channels)
        self.pconv_act = nn.ReLU(inplace=True)

        # 新增：RepLK 注意力模块
        self.attention = RepLKBlock(
            mid_channels, 
            mid_channels,
            27, 5, 0.0, False
        )

        # 第三步：后处理卷积
        self.post_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 预处理卷积
        x = self.pre_conv(x)
        # PConv 处理
        x = self.pconv_act(self.pconv_bn(self.pconv(x)))
        # RepLK 注意力增强
        x = self.attention(x)
        # 后处理卷积
        x = self.post_conv(x)
        return x

class Down_PConv_RepLK(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_PConv_RepLK(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
######################### PConv、RepLKBlock加到模型里 开始 ##################

######################### PConv、RepLKBlock加到模型里 结束 ##################

######################### ODConv2d加到DoubleConv 开始 ##################
class DoubleConv_ODConv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = ODConv2d(in_channels, mid_channels,kernel_size=3) 
        self.conv2 = ODConv2d(mid_channels, out_channels,kernel_size=3)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Down_ODConv2d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_ODConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

######################### ODConv2d加到DoubleConv 结束 ##################

######################### ScConv加到DoubleConv 开始 ##################
class DoubleConv_ScConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        #self.conv1 = Conv(in_channels,mid_channels)
        self.conv1 = Conv(in_channels, mid_channels)
        self.conv2 = ScConv(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down_ScConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_ScConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
######################### ScConv加到DoubleConv 结束 ##################

######################### DualConv加到DoubleConv 开始 ##################

class DoubleConv_DualConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = DualConv(in_channels,mid_channels) 
        self.conv2 = DualConv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Down_DualConv(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_DualConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### DualConv加到DoubleConv 结束 ##################


######################### RFAConv加到DoubleConv 开始 ##################
class DoubleConv_RFAConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        #self.conv1 = Conv(in_channels,mid_channels)
        self.conv1 = RFAConv(in_channels, mid_channels,kernel_size=3)
        self.conv2 = RFAConv(mid_channels, out_channels,kernel_size=3)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Down_RFAConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_RFAConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
######################### RFAConv加到DoubleConv 结束 ##################

######################### AKConv加到DoubleConv 开始 ##################

class DoubleConv_AKConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = AKConv(in_channels, mid_channels, 1) 
        self.conv2 = AKConv(mid_channels, out_channels,1)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down_AKConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_AKConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


######################### AKConv加到DoubleConv 结束 ##################


######################### FastKAN加到DoubleConv 开始 ##################

 
class DoubleConv_FastKAN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(in_channels,mid_channels)
        self.attention = FastKANConv2DLayer(mid_channels, mid_channels, kernel_size=3, padding=3 // 2)
        self.conv2 = Conv(mid_channels, out_channels)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class Down_FastKAN(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_FastKAN(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
######################### FastKAN加到DoubleConv 结束 ##################

######################### ADown加到RepLKBlockConv 开始 ##################  
class DoubleConv_ADown_RepLKBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.min_spatial_size = 2
        
        # 添加一个标准卷积作为备选
        self.standard_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.standard_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 原有的ADown和注意力模块
        self.conv1 = ADown(in_channels, mid_channels, self.min_spatial_size)
        self.attention = RepLKBlock(mid_channels, mid_channels, 27, 5, 0.0, False)
        self.conv2 = ADown(mid_channels, out_channels, self.min_spatial_size)
        
    def forward(self, x):
        try:
            # 如果输入尺寸太小，使用标准卷积
            if x.size(-1) <= self.min_spatial_size * 2 or x.size(-2) <= self.min_spatial_size * 2:
                x = self.standard_conv1(x)
                x = self.standard_conv2(x)
                return x
            
            # 否则使用ADown和注意力机制
            x = self.conv1(x)
            
            if x.size(-1) > self.min_spatial_size and x.size(-2) > self.min_spatial_size:
                identity = x
                x = self.attention(x)
                x = x + identity
            
            if x.size(-1) > self.min_spatial_size and x.size(-2) > self.min_spatial_size:
                x = self.conv2(x)
            
            return x
            
        except Exception as e:
            raise

class Down_ADown_RepLKBlockConv(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.min_spatial_size = 2
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv_ADown_RepLKBlockConv(in_channels, out_channels)
 
    def forward(self, x):
        try:
            # 检查输入尺寸
            if x.size(-1) <= self.min_spatial_size or x.size(-2) <= self.min_spatial_size:
                return self.conv(x)  # 直接进行卷积，跳过maxpool
            
            # 计算maxpool后的尺寸
            out_h = x.size(-2) // 2
            out_w = x.size(-1) // 2
            
            if out_h < self.min_spatial_size or out_w < self.min_spatial_size:
                return self.conv(x)
                
            x = self.maxpool(x)
            x = self.conv(x)
            
            return x
            
        except Exception as e:
            raise

######################### ADown加到RepLKBlockConv 结束 ##################  