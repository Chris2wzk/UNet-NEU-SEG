import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class ADown(nn.Module):
    """Asymmetric Downsampling module.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    def __init__(self, c1, c2, min_spatial_size=2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, self.c, 3, 2, 1)
        self.cv2 = Conv(c1, self.c, 1, 1, 0)
        self.min_spatial_size = min_spatial_size
        
    
    def forward(self, x):
        try:
            
            # 如果特征图太小，跳过下采样
            if x.size(-1) <= self.min_spatial_size or x.size(-2) <= self.min_spatial_size:
                return x
            
            # 计算下采样后的尺寸
            out_h = (x.size(-2) + 1) // 2
            out_w = (x.size(-1) + 1) // 2
            
            # 如果下采样后尺寸太小，也跳过
            if out_h < self.min_spatial_size or out_w < self.min_spatial_size:
                print(f"[WARNING] Output would be too small: {out_h}x{out_w}, skipping downsampling")
                return x
            
            # 正常处理
            x1 = self.cv1(x)
            x2 = F.max_pool2d(x, 3, 2, 1)
            x2 = self.cv2(x2)
            
            # 确保通道数匹配
            assert x1.shape[1] == x2.shape[1], \
                f"Channel mismatch: {x1.shape[1]} vs {x2.shape[1]}"
            
            out = torch.cat([x1, x2], dim=1)
            
            return out
            
        except Exception as e:
            print(f"[ERROR] Error in ADown.forward: {str(e)}")
            print(f"[ERROR] Input shape: {x.shape}")
            raise