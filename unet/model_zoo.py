from .unet_model import *

MODEL_ZOO = {
    'UNet': UNet,
    'UNetRepLKBlock_Conv':UNetRepLKBlock_Conv,
    'UNetFastKAN':UNetFastKAN,
    'UNet_MobileNet': UNet_MobileNet,
    'UNet_MobileNet_Rep': UNet_MobileNet_Rep,
    'UNet_MobileNetV4': UNet_MobileNetV4,
    'ResUNet':ResUNet,
    'UNet_RepLKBlockConv_FastKAN':UNet_RepLKBlockConv_FastKAN,
    'UNet_ADown_RepLKBlockConv_FastKAN':UNet_ADown_RepLKBlockConv_FastKAN,
    'UNet_PConv_RepLKBlock_FastKAN': UNet_PConv_RepLKBlock_FastKAN,
    'UNetCBAM':UNetCBAM,
    'UNetSimAM':UNetSimAM,
    'UNetShuffleAttention':UNetShuffleAttention
}
