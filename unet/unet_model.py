from blocks import *  
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### 基线模型 结束 #########################



######################### iRMB加到卷积层 开始 ##############################
#在卷积层中添加注意力机制
class UNetiRMB_Conv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetiRMB_Conv, self).__init__()
        self.model_name = 'UNetiRMB_Conv'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        self.inc = (DoubleConvAttentioniRMB(n_channels, 64))
        self.down1 = (DownAttentioniRMB(64, 128))
        self.down2 = (DownAttentioniRMB(128, 256))
        self.down3 = (DownAttentioniRMB(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DownAttentioniRMB(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
#如果torch版本环境较低，请注释掉这一段和train.py第314行的model.use_checkpointing()
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### iRMB加到卷积层 结束 ##############################

######################### CBAM加到连接处 开始 ##############################

#在连接处添加注意力机制
class UNetCBAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNetCBAM, self).__init__()
 
        self.model_name = 'UNetCBAM'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = CBAM(64)
            self.attention2 = CBAM(128)
            self.attention3 = CBAM(256)
            self.attention4 = CBAM(512)
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### CBAM加到连接处 结束 ##############################

######################### SimAM加到连接处 开始 ##############################
#在连接处添加注意力机制
class UNetSimAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNetSimAM, self).__init__()
 
        self.model_name = 'UNetSimAM'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:

            self.attention1 = SimAM()
            self.attention2 = SimAM()
            self.attention3 = SimAM()
            self.attention4 = SimAM()
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
 


######################### SimAM加到连接处 结束 ##############################

######################### shuffleAtt加到连接处 开始 ##############################

#在连接处添加注意力机制
class UNetShuffleAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNetShuffleAttention, self).__init__()
 
        self.model_name = 'UNetShuffleAttention'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = ShuffleAttention(64)
            self.attention2 = ShuffleAttention(128)
            self.attention3 = ShuffleAttention(256)
            self.attention4 = ShuffleAttention(512)
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### shuffleAtt加到连接处 结束 ##############################



######################### RepLKBlock加到连接处 开始 ##############################
#在连接处添加注意力机制
class UNetRepLKBlock(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNetRepLKBlock, self).__init__()
 
        self.model_name = 'UNetRepLKBlock'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = RepLKBlock(64,64,27, 5, 0.0, False)
            self.attention2 = RepLKBlock(128,128,27, 5, 0.0, False)
            self.attention3 = RepLKBlock(256,256,27, 5, 0.0, False)
            self.attention4 = RepLKBlock(512,512,27, 5, 0.0, False)
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### RepLKBlock加到连接处 结束 ##############################

######################### RepLKBlock加到卷积层 开始 ##############################

class UNetRepLKBlock_Conv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetRepLKBlock_Conv, self).__init__()
        self.model_name = 'UNetRepLKBlock_Conv'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
 
        self.inc = (DoubleConvAttentionRepLK(n_channels, 64))
        self.down1 = (DownAttentionRepLK(64, 128))
        self.down2 = (DownAttentionRepLK(128, 256))
        self.down3 = (DownAttentionRepLK(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DownAttentionRepLK(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
#如果torch版本环境较低，请注释掉这一段和train.py第314行的model.use_checkpointing()
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### RepLKBlock加到卷积层 结束 ##############################


######################### FastKANConvNDLayer加到连接处 开始 ###################
#在连接处添加注意力机制
class UNetFastKAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNetFastKAN, self).__init__()
 
        self.model_name = 'UNetFastKAN'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = FastKANConv2DLayer(64, 64, kernel_size=3, padding=3 // 2)
            self.attention2 = FastKANConv2DLayer(128, 128, kernel_size=3, padding=3 // 2)
            self.attention3 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=3 // 2)
            self.attention4 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=3 // 2)
 
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### FastKANConvNDLayer加到连接处 结束 ###################

######################### PConv、RepLKBlock改进DoubleConv 开始 ###################

class UNet_PConv_RepLKBlockConv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_PConv_RepLKBlockConv, self).__init__()
        self.model_name = 'UNet_PConv_RepLKBlockConv'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv_PConv_RepLK(n_channels, 64))
        self.down1 = (Down_PConv_RepLK(64, 128))
        self.down2 = (Down_PConv_RepLK(128, 256))
        self.down3 = (Down_PConv_RepLK(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down_PConv(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### PConv、RepLKBlock改进DoubleConv 结束 ###################


######################### PConv、RepLKBlockConv、FastKAN加到模型里 开始 ##############################

#在连接处添加注意力机制
class UNet_PConv_RepLKBlock_FastKAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_PConv_RepLKBlock_FastKAN, self).__init__()
 
        self.model_name = 'UNet_PConv_RepLKBlock_FastKAN'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv_PConv_RepLK(n_channels, 64))
        self.down1 = (Down_PConv_RepLK(64, 128))
        self.down2 = (Down_PConv_RepLK(128, 256))
        self.down3 = (Down_PConv_RepLK(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down_PConv_RepLK(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = FastKANConv2DLayer(64, 64, kernel_size=3, padding=3 // 2)
            self.attention2 = FastKANConv2DLayer(128, 128, kernel_size=3, padding=3 // 2)
            self.attention3 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=3 // 2)
            self.attention4 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=3 // 2)
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### PConv、RepLKBlockConv、FastKAN加到模型里 结束 ##################

######################### PConv、RepLKBlock加到连接处 开始 #########

class UNet_PConv_RepLKBlock(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_PConv_RepLKBlock, self).__init__()
 
        self.model_name = 'UNet_PConv_RepLKBlock'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv_PConv(n_channels, 64))
        self.down1 = (Down_PConv(64, 128))
        self.down2 = (Down_PConv(128, 256))
        self.down3 = (Down_PConv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down_PConv(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = RepLKBlock(64,64,27, 5, 0.0, False)
            self.attention2 = RepLKBlock(128,128,27, 5, 0.0, False)
            self.attention3 = RepLKBlock(256,256,27, 5, 0.0, False)
            self.attention4 = RepLKBlock(512,512,27, 5, 0.0, False)
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### PConv、RepLKBlock加到连接处 结束 #########


######################### PConv、FastKAN加到模型里 开始 #########
#在连接处添加注意力机制
class UNet_PConv_FastKAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_PConv_FastKAN, self).__init__()
 
        self.model_name = 'UNet_PConv_FastKAN'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConv_PConv(n_channels, 64))
        self.down1 = (Down_PConv(64, 128))
        self.down2 = (Down_PConv(128, 256))
        self.down3 = (Down_PConv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down_PConv(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = FastKANConv2DLayer(64, 64, kernel_size=3, padding=3 // 2)
            self.attention2 = FastKANConv2DLayer(128, 128, kernel_size=3, padding=3 // 2)
            self.attention3 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=3 // 2)
            self.attention4 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=3 // 2)
 
 
 
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
######################### PConv、FastKANet加到模型里 结束 #########


######################### RepLKBlockConv、FastKAN加到模型里 开始 ##############################

#在连接处添加注意力机制
class UNet_RepLKBlockConv_FastKAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_RepLKBlockConv_FastKAN, self).__init__()
 
        self.model_name = 'UNet_RepLKBlockConv_FastKAN'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        self.inc = (DoubleConvAttentionRepLK(n_channels, 64))
        self.down1 = (DownAttentionRepLK(64, 128))
        self.down2 = (DownAttentionRepLK(128, 256))
        self.down3 = (DownAttentionRepLK(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DownAttentionRepLK(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
 
        if self.attention:
            self.attention1 = FastKANConv2DLayer(64, 64, kernel_size=3, padding=3 // 2)
            self.attention2 = FastKANConv2DLayer(128, 128, kernel_size=3, padding=3 // 2)
            self.attention3 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=3 // 2)
            self.attention4 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=3 // 2)
 
    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.attention1(x1) + x1
 
        x2 = self.down1(x1)
        if self.attention:
            x2 = self.attention2(x2) + x2
 
        x3 = self.down2(x2)
        if self.attention:
            x3 = self.attention3(x3) + x3
 
        x4 = self.down3(x3)
        if self.attention:
            x4 = self.attention4(x4) + x4
 
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### ADown、RepLKBlock_Conv、FastKAN加到模型里 结束 ##############################

class UNet_ADown_RepLKBlockConv_FastKAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(UNet_ADown_RepLKBlockConv_FastKAN, self).__init__()
 
        self.model_name = 'UNet_ADown_RepLKBlockConv_FastKAN'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
 
        # Encoder path
        self.inc = (DoubleConv_ADown_RepLKBlockConv(n_channels, 64))
        self.down1 = (Down_ADown_RepLKBlockConv(64, 128))
        self.down2 = (Down_ADown_RepLKBlockConv(128, 256))
        self.down3 = (Down_ADown_RepLKBlockConv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down_ADown_RepLKBlockConv(512, 1024 // factor))
        
        # Decoder path with bilinear upsampling
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        
        # 最终输出层
        self.outc = (OutConv(64, n_classes))
        
        # 添加额外的上采样层确保输出尺寸匹配
        self.final_upsample = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)
 
        if self.attention:
            self.attention1 = FastKANConv2DLayer(64, 64, kernel_size=3, padding=3 // 2)
            self.attention2 = FastKANConv2DLayer(128, 128, kernel_size=3, padding=3 // 2)
            self.attention3 = FastKANConv2DLayer(256, 256, kernel_size=3, padding=3 // 2)
            self.attention4 = FastKANConv2DLayer(512, 512, kernel_size=3, padding=3 // 2)
 
    def forward(self, x):
        input_size = x.shape[-2:]  # 保存输入尺寸
        
        # Encoder path
        x1 = self.inc(x)
        if self.attention and x1.size(-1) > 2 and x1.size(-2) > 2:
            x1 = self.attention1(x1) + x1
        
        x2 = self.down1(x1)
        if self.attention and x2.size(-1) > 2 and x2.size(-2) > 2:
            x2 = self.attention2(x2) + x2
        
        x3 = self.down2(x2)
        if self.attention and x3.size(-1) > 2 and x3.size(-2) > 2:
            x3 = self.attention3(x3) + x3
        
        x4 = self.down3(x3)
        if self.attention and x4.size(-1) > 2 and x4.size(-2) > 2:
            x4 = self.attention4(x4) + x4
        
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 通过1x1卷积得到所需的通道数
        x = self.outc(x)
        
        # 确保输出尺寸与输入匹配
        if x.shape[-2:] != input_size:
            x = self.final_upsample(x)
        
        return x
 
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

######################### ADown\RepLKBlock_Conv、FastKAN加到模型里 结束 ##############################

class UNet_MobileNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_MobileNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = 'UNet_MobileNet'
        # MobileNet backbone
        self.backbone = MobileNet(n_channels)

        # 上采样和卷积层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 通过backbone提取特征
        x2, x1, x0 = self.backbone(x)  # x2:104x104x256, x1:52x52x512, x0:26x26x1024

        # 上采样路径
        p5 = self.up1(x0)  # 上采样到和x1相同的尺寸
        # 确保p5和x1尺寸匹配
        if p5.size()[2:] != x1.size()[2:]:
            p5 = F.interpolate(p5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        p5 = self.conv1(p5)
        
        p4 = torch.cat([x1, p5], dim=1)
        p4 = self.up2(p4)
        # 确保p4和x2尺寸匹配
        if p4.size()[2:] != x2.size()[2:]:
            p4 = F.interpolate(p4, size=x2.size()[2:], mode='bilinear', align_corners=True)
        p4 = self.conv2(p4)
        
        p3 = torch.cat([x2, p4], dim=1)
        p3 = self.up3(p3)
        p3 = self.conv3(p3)
        
        p3 = self.up4(p3)
        p3 = self.conv4(p3)
        
        logits = self.outc(p3)

        return logits

    def use_checkpointing(self):
        self.backbone = torch.utils.checkpoint.checkpoint_sequential([
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3
        ], 3, self.backbone(x))

class UNet_MobileNet_Rep(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_MobileNet_Rep, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = 'UNet_MobileNet_Rep'
        # self.attention = attention

        # 使用增强版的MobileNet作为backbone
        self.backbone = MobileNet_Rep(n_channels)

        # 上采样和卷积层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 通过backbone提取特征
        x2, x1, x0 = self.backbone(x)  # x2:104x104x256, x1:52x52x512, x0:26x26x1024

        # 上采样路径
        p5 = self.up1(x0)  # 上采样到和x1相同的尺寸
        # 确保p5和x1尺寸匹配
        if p5.size()[2:] != x1.size()[2:]:
            p5 = F.interpolate(p5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        p5 = self.conv1(p5)           # 26x26x512
        
        p4 = torch.cat([x1, p5], dim=1)   # 26x26x1024
        p4 = self.up2(p4)             
        p4 = self.conv2(p4)           # 52x52x256
        
        p3 = torch.cat([x2, p4], axis=1)  # 52x52x512
        p3 = self.up3(p3)
        p3 = self.conv3(p3)           # 104x104x128
        
        p3 = self.up4(p3)
        p3 = self.conv4(p3)           # 208x208x64
        
        logits = self.outc(p3)        # 208x208xn_classes

        return logits

    def use_checkpointing(self):
        self.backbone = torch.utils.checkpoint.checkpoint_sequential([
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3
        ], 3, self.backbone(x))

class UNet_MobileNetV4(nn.Module):
    def __init__(self, n_channels, n_classes, model_name="MNV4HybridMedium", bilinear=False):
        super(UNet_MobileNetV4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = 'UNet_MobileNetV4'

        # 使用MobileNetV4作为backbone
        self.backbone = create_mobilenetv4_backbone(model_name=model_name, n_channels=n_channels)

        # 获取backbone的输出通道数
        out_channels = self.backbone.out_channels
        
        # 上采样和卷积层
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(out_channels["layer4"], out_channels["layer3"])

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(out_channels["layer3"]*2, out_channels["layer2"])

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(out_channels["layer2"]*2, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 通过backbone提取特征
        x2, x1, x0 = self.backbone(x)  # x2:浅层特征, x1:中层特征, x0:深层特征

        # 上采样路径
        p5 = self.up1(x0)  # 上采样到和x1相同的尺寸
        # 确保p5和x1尺寸匹配
        if p5.size()[2:] != x1.size()[2:]:
            p5 = F.interpolate(p5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        p5 = self.conv1(p5)
        
        p4 = torch.cat([x1, p5], dim=1)
        p4 = self.up2(p4)
        p4 = self.conv2(p4)
        
        p3 = torch.cat([x2, p4], dim=1)
        p3 = self.up3(p3)
        p3 = self.conv3(p3)
        
        p3 = self.up4(p3)
        p3 = self.conv4(p3)
        
        logits = self.outc(p3)

        return logits

    def use_checkpointing(self):
        self.backbone = torch.utils.checkpoint.checkpoint_sequential([
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ], 4, self.backbone(x))

######################### ResNet Block 开始 ###################

class BasicBlock(nn.Module):          
    expansion = 1
    '''
    expansion通道扩充比例
    out_channels就是输出的channel
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    expansion = 4
    
    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''
    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out

######################### ResNet Block 结束 ###################

######################### ResUNet 开始 ###################

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, attention=False):
        super(ResUNet, self).__init__()
        
        self.model_name = 'ResUNet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        
        # 基础配置
        nb_filter = [64, 256, 512, 1024, 2048]  # 由于Bottleneck的expansion=4，所以通道数要相应调整
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder路径 - ResNet50结构
        self.inc = VGGBlock(n_channels, 64, 64)  # 第一层保持不变
        self.down1 = self._make_encoder_layer(BottleNeck, 64, 64, 3)  # 64*4=256
        self.down2 = self._make_encoder_layer(BottleNeck, 256, 128, 4)  # 128*4=512
        self.down3 = self._make_encoder_layer(BottleNeck, 512, 256, 6)  # 256*4=1024
        self.down4 = self._make_encoder_layer(BottleNeck, 1024, 512, 3)  # 512*4=2048
        
        # Decoder路径
        factor = 2 if bilinear else 1
        self.up1 = VGGBlock((nb_filter[3] + nb_filter[4])//factor, nb_filter[3]//4, 
                           nb_filter[3])  # 考虑Bottleneck的expansion=4
        self.up2 = VGGBlock((nb_filter[2] + nb_filter[3])//factor, nb_filter[2]//4, 
                           nb_filter[2])
        self.up3 = VGGBlock((nb_filter[1] + nb_filter[2])//factor, nb_filter[1]//4, 
                           nb_filter[1])
        self.up4 = VGGBlock(64 + nb_filter[1], 64, 64)  # 最后一层保持64通道
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _make_encoder_layer(self, block, in_channels, middle_channels, num_blocks):
        layers = []
        # 第一个block可能需要调整维度
        layers.append(block(in_channels, middle_channels))
        # 后续blocks输入通道数已经是middle_channels * 4
        for _ in range(1, num_blocks):
            layers.append(block(middle_channels * block.expansion, middle_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x5 = self.down4(self.pool(x4))
        
        # Decoder with skip connections
        x5_up = self.up(x5)
        # 确保x5_up和x4尺寸匹配
        if x5_up.size()[2:] != x4.size()[2:]:
            x5_up = F.interpolate(x5_up, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = self.up1(torch.cat([x4, x5_up], 1))
        
        x_up = self.up(x)
        # 确保x_up和x3尺寸匹配
        if x_up.size()[2:] != x3.size()[2:]:
            x_up = F.interpolate(x_up, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = self.up2(torch.cat([x3, x_up], 1))
        
        x_up = self.up(x)
        # 确保x_up和x2尺寸匹配
        if x_up.size()[2:] != x2.size()[2:]:
            x_up = F.interpolate(x_up, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = self.up3(torch.cat([x2, x_up], 1))
        
        x_up = self.up(x)
        # 确保x_up和x1尺寸匹配
        if x_up.size()[2:] != x1.size()[2:]:
            x_up = F.interpolate(x_up, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = self.up4(torch.cat([x1, x_up], 1))
        
        logits = self.outc(x)
        return logits
        
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)

######################### ResUNet 结束 ###################



