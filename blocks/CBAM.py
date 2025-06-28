import torch
import torch.nn as nn
 
########CBAM
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
 
 
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
 
 
class CBAM(nn.Module):
    def __init__(self, channel, c_attention=True, s_attention=True):
        super(CBAM, self).__init__()
        self.c_attention = c_attention
        self.s_attention = s_attention
        if self.c_attention:
            self.channel_attention = ChannelAttention(channel)
        if self.s_attention:
            self.spatial_attention = SpatialAttention()
 
    def forward(self, x):
        out = x
        if self.c_attention:
            out = self.channel_attention(out) * out
        if self.s_attention:
            out = self.spatial_attention(out) * out
        return out
##############CBAM
 