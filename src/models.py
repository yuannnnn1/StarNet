"""CIFAR adaptation of Ma et al. (CVPR 2024); see docs/research.md."""
import torch
from torch import nn


def convbn(cin, cout, kernel=1, stride=1, groups=1, bn=True):
    layers = [nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=True)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    return nn.Sequential(*layers)


class StarBlock(nn.Module):
    def __init__(self, width, operation="multiply"):
        super().__init__()
        self.operation = operation
        self.dw1 = convbn(width, width, 7, groups=width)
        self.f1 = convbn(width, width * 3, bn=False)
        self.f2 = convbn(width, width * 3, bn=False)
        self.project = convbn(width * 3, width)
        self.dw2 = convbn(width, width, 7, groups=width, bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        z = self.dw1(x)
        a, b = self.act(self.f1(z)), self.f2(z)
        z = a * b if self.operation == "multiply" else a + b
        return x + self.dw2(self.project(z))


class StarNet(nn.Module):
    def __init__(self, operation="multiply"):
        super().__init__()
        self.stem = nn.Sequential(convbn(3, 24, 3), nn.ReLU6())
        layers, cin = [], 24
        for width, depth, stride in zip([24, 48, 96], [1, 1, 2], [1, 2, 2]):
            layers.append(convbn(cin, width, 3, stride))
            layers.extend(StarBlock(width, operation) for _ in range(depth))
            cin = width
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm2d(96), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 10))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, cin = [], 3
        for width in [32, 64, 128]:
            layers.extend([convbn(cin, width, 3), nn.ReLU(), convbn(width, width, 3), nn.ReLU(), nn.MaxPool2d(2)])
            cin = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def build_model(name):
    if name == "baseline":
        return BaselineCNN()
    if name in ("starnet", "ablation"):
        return StarNet("multiply" if name == "starnet" else "add")
    raise ValueError(name)


def compute_macs(model):
    """Conv/linear multiply-accumulate count, excluding BN/activation/pooling."""
    total, hooks = [0], []
    def count(layer, inputs, output):
        if isinstance(layer, nn.Conv2d):
            total[0] += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            total[0] += output.numel() * layer.in_features
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(count))
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
    for hook in hooks:
        hook.remove()
    return total[0]
