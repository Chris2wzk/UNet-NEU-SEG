import torch
import torch.nn as nn
from timm.layers import CondConv2d
from torch import Tensor
import torch.nn.functional as F
class PConv(nn.Module):
    def __init__(self,
                 dim: int,
                 n_div: int,
                 forward: str = "split_cat",
                 kernel_size: int = 1) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
 
        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)
 
        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
 
    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim.conv, :, :] = self.conv(x[:, :self.dim.conv, :, :])
        return x
 
    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), 1)
 
        return x
 
###################################PConv######################
 