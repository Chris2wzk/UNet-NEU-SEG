import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import init
from einops import rearrange

# =============================================================================
# 针对 "Unable to find a valid cuDNN algorithm to run convolution" 的常见解决手段
# （以下设置仅限本文件，若仍无效，需考虑降batch_size、升级显卡驱动/CUDA/cuDNN等）
# =============================================================================

# 1) 允许更大范围的算法搜索
cudnn.benchmark = True

# 2) 关闭 cudnn 的确定性算法，以便它有更多可用的优化实现
cudnn.deterministic = False

# 3) 允许 TF32（如果硬件支持），进一步放宽卷积实现
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 4) 允许增大 cuDNN 占用内存上限（根据机器显存情况自行调整）
os.environ["CUDNN_CONV_MAX_WORKSPACE_SIZE_BYTES"] = "1073741824"  # 1GB

######################################## Common ########################################
class Conv(nn.Module):
    default_act = nn.ReLU()  # 默认激活函数
    
    def __init__(self, c1, c2, k=1, s=1, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x):
        # 对输入进行 contiguous() 强制，避免由于内存布局造成的算法不可用
        x = x.contiguous()
        return self.act(self.bn(self.conv(x)))

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or (self.drop_prob is None) or (self.drop_prob == 0.0):
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

######################################## SE ########################################
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.contiguous()  # 同样强制 contiguous
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

######################################## iRMB ########################################
class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s

        if self.attn_s:
            # 空间注意力
            assert dim_in % dim_head == 0, 'dim_in必须能被dim_head整除'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre

            self.qk = nn.Conv2d(dim_in, dim_in * 2, kernel_size=1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1,
                          groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            # 不用空间注意力时，仅做 v 投影
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1,
                              groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()

        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        # 强制变为 contiguous，防止打乱布局导致的问题
        x = x.contiguous()
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape

        if self.attn_s:
            # 根据 window_size 对 x 进行 padding
            window_size_W = self.window_size if self.window_size > 0 else W
            window_size_H = self.window_size if self.window_size > 0 else H
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H

            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b), mode="constant", value=0)

            n1 = (H + pad_b) // window_size_H
            n2 = (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2)
            x = x.contiguous()

            # 计算 QK
            qk = self.qk(x).contiguous()
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head',
                           qk=2, heads=self.num_head, dim_head=self.dim_head)
            q, k = qk[0], qk[1]

            # 空间注意力矩阵
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)

            # 先注意力后投影 or 先投影后注意力
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head)
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w',
                                  heads=self.num_head, h=x.shape[-2], w=x.shape[-1])
                x_spa = x_spa.contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x).contiguous()
                v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.num_head)
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) d -> b (heads d) h w',
                                  heads=self.num_head, h=x.shape[-2], w=x.shape[-1])
                x_spa = x_spa.contiguous()

            # 恢复回原大小
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2)
            x = x.contiguous()

            # 如果 padding 过，需要裁切掉
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x).contiguous()

        # 通局部深度卷积+SE
        local_out = self.conv_local(x).contiguous()
        local_out = self.se(local_out).contiguous()
        if self.has_skip:
            x = x + local_out
        else:
            x = local_out

        # 最终projection
        x = self.proj_drop(x)
        x = x.contiguous()
        x = self.proj(x).contiguous()

        # 残差连接
        if self.has_skip:
            x = shortcut + self.drop_path(x)
        return x
 
 
 
######################################## iRMB  ########################################