import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from collections import OrderedDict
import operator
from itertools import islice

_pair = _ntuple(1)

__all__ = [
    'resnet18gn', 'resnet34gn', 'resnet50gn', 'resnet101gn',
    'resnet152gn'
]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


class Conv1d(_ConvNd):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv1d, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     False,
                                     _pair(0),
                                     groups,
                                     bias,
                                     padding_mode='zeros')

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(
                OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2


def resnet18gn(input_dim, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GNResNet(input_dim, BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34gn(input_dim, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GNResNet(input_dim, BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50gn(input_dim, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GNResNet(input_dim, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101gn(input_dim, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GNResNet(input_dim, Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152gn(input_dim, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GNResNet(input_dim, Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


class GNResNet(nn.Module):

    def __init__(self,
                 input_dim,
                 block,
                 layers,
                 in_features=256,
                 num_classes=1000,
                 no_linear=True):
        self.input_dim = input_dim
        self.inplanes = 64
        self.in_features = in_features
        self.num_classes = num_classes
        super(GNResNet, self).__init__()
        self.conv1 = nn.Conv1d(self.input_dim,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=True)
        # self.bn1 = DomainSpecificBatchNorm1d(64, self.num_domains)
        self.bn1 = nn.GroupNorm(32, 64, eps=1e-5, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2)
        self.no_linear = no_linear
        if not self.no_linear:
            # self.avgpool = nn.AvgPool1d(7, stride=1)
            if self.in_features != 0:
                self.fc1 = nn.Linear(512 * block.expansion, self.in_features)
                self.fc2 = nn.Linear(self.in_features, num_classes)
            else:
                self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1d(self.inplanes,
                       planes * block.expansion,
                       kernel_size=1,
                       stride=stride,
                       bias=True),
                # DomainSpecificBatchNorm1d(planes * block.expansion,
                                        #   num_domains),
                nn.GroupNorm(32, planes * block.expansion, eps=1e-5, affine=True)
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, with_ft=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if not self.no_linear:
            x = x.mean(3).mean(2)  # global average pooling
            x = x.view(x.size(0), -1)
            if self.in_features != 0:
                x = self.fc1(x)
                feat = x
                x = self.fc2(x)
            else:
                x = self.fc(x)
                feat = x
            if with_ft:
                return x, feat
            else:
                return x
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = DomainSpecificBatchNorm1d(planes, num_domains)
        self.bn1 = nn.GroupNorm(32, planes, eps=1e-5, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = DomainSpecificBatchNorm1d(planes, num_domains)
        self.bn2 = nn.GroupNorm(32, planes, eps=1e-5, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        # self.bn1 = DomainSpecificBatchNorm1d(planes, num_domains)
        self.bn1 = nn.GroupNorm(32, planes, eps=1e-5, affine=True)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=True)
        # self.bn2 = DomainSpecificBatchNorm1d(planes, num_domains)
        self.bn2 = nn.GroupNorm(32, planes, eps=1e-5, affine=True)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=True)
        # self.bn3 = DomainSpecificBatchNorm1d(planes * 4, num_domains)
        self.bn3 = nn.GroupNorm(32, planes*4, eps=1e-5, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
