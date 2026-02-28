import torch
from torch import nn
import torch.nn.functional as F


# -------------------------
# Helpers
# -------------------------
def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
    Args:
        k: kernel size
        p: padding (if None, auto-compute)
        d: dilation
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def Upsample(x, size, align_corners=False):
    """Wrapper around interpolate (kept name for compatibility)."""
    return F.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


# -------------------------
# Basic blocks
# -------------------------
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# -------------------------
# SCRG
# -------------------------
class SCRG(nn.Module):
    def __init__(self, inc, input_dim=64):
        super().__init__()
        self.input_dim = input_dim

        # keep attribute names identical for state_dict compatibility
        self.d_in1 = Conv(input_dim // 2, input_dim // 2, 1)
        self.d_in2 = Conv(input_dim // 2, input_dim // 2, 1)

        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim // 2, kernel_size=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        # keep original tensor update order to preserve exact numerics
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(
            g_H_feature * H_feature, size=L_feature.size()[2:], align_corners=False
        )
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(
            g_L_feature * L_feature, size=H_feature.size()[2:], align_corners=False
        )

        H_feature = Upsample(H_feature, size=L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out