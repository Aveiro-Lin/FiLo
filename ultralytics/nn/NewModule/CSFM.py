import torch
import torch.nn as nn
import numpy as np

__all__ = ["CSFM", "RepNCSPELAN4_high"]


# -------------------------
# Common helpers
# -------------------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs (matches original behavior)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard conv+BN+SiLU (matches original behavior)."""
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
# Rep-style blocks
# -------------------------
class RepConvN(nn.Module):
    """RepConv (training-time multi-branch; optional deploy fusion)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # keep identical attributes for state_dict compatibility
        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x):
        """Forward process for fused deploy conv."""
        return self.act(self.conv(x))

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn_tensor(self.conv1)
        k1, b1 = self._fuse_bn_tensor(self.conv2)
        kid, bid = self._fuse_bn_tensor(self.bn)
        return k3 + self._pad_1x1_to_3x3_tensor(k1) + kid, b3 + b1 + bid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size**2
        return k

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            raise TypeError(f"Unsupported branch type: {type(branch)}")

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Create self.conv and delete training-time branches (same as original)."""
        if hasattr(self, "conv"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)

        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        for para in self.parameters():
            para.detach_()

        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class RepNBottleneck(nn.Module):
    """Standard bottleneck (RepConvN -> Conv), identical topology."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class RepNCSP(nn.Module):
    """CSP bottleneck with RepNBottleneck blocks (same as original)."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# -------------------------
# FiLo blocks
# -------------------------
class CSFM(nn.Module):
    """CSFM: CSP-ELAN style block (matches original behavior)."""
    def __init__(self, c1, c2, c5=1):
        super().__init__()
        c3 = int(c2 / 2)
        c4 = int(c3 / 2)
        self.c = c3 // 2

        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv3 = nn.Sequential(RepNCSP(c3, c3, c5))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        tmp = self.cv1(x)
        tmp3 = self.cv3(tmp)
        parts = list(tmp.chunk(2, 1))
        parts.append(tmp3)
        return self.cv4(torch.cat(parts, 1))

    def forward_split(self, x):
        # keep identical behavior (this method references self.cv2 in the original; unused by FiLo YAML)
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN4_high(nn.Module):
    """ELAN4-high style block (matches original behavior)."""
    def __init__(self, c1, c2, c5=1):
        super().__init__()
        c3 = c2
        c4 = int(c3 / 2)
        self.c = c3 // 2

        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


if __name__ == "__main__":
    image_size = (1, 24, 224, 224)
    image = torch.rand(*image_size)
    model = CSFM(24, 24)
    out = model(image)
    print(out.size())