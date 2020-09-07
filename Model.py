# script for submission

import re
import random
import torch
import torchvision
import numpy as np
import scipy.signal as ss
import scipy.ndimage as snd
import skimage.transform as skt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from collections import OrderedDict
from PIL import Image

from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 ctonly=False,
                 moco=True):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
        self.ctonly = ctonly

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        if moco:
            features = [ConvBNReLU(2, input_channel, stride=2, norm_layer=norm_layer)]
        else:
            features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier_new = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier_new(x)
        return x

    def forward(self, x, lungseg=None):
        if not self.ctonly:
            x = torch.cat([x, lungseg], dim=1)
        return self._forward_impl(x)



# DenseNet
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, ctonly=False, 
                 moco=True):

        super(DenseNet, self).__init__()
        self.ctonly = ctonly

        # First convolution
        ninputplane = 2 if moco else 3
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(ninputplane, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier_new = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, lungseg=None):
        if not self.ctonly:
            x = torch.cat([x, lungseg], dim=1)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier_new(out)
        return out

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def densenet121(task, progress = True, moco=False, ctonly=False, **kwargs):
    kwargs['num_classes'] = 128 # moco_dim
    if 'pretrained' in kwargs:
        moco = False # using pretrained model
        del kwargs['pretrained']

    net = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, ctonly=ctonly, moco=moco, **kwargs)
    if moco:
        state = torch.load("./chpt/moco_densenet.pth")["state_dict"]
        state_dict = {k.replace('encoder_q.',''):state[k] for k in state.keys() if 'encoder_q' in k}
    else:
        state_dict = _load_state_dict(net, model_urls["densenet121"], progress)
    
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    if task in ['classification']:
        net.classifier_new = torch.nn.Linear(1024, 1)

    if ctonly:
        net.features[0].weight = torch.nn.Parameter(net.features[0].weight[:,:1,:,:])
    else:
        net.features[0].weight = torch.nn.Parameter(net.features[0].weight[:,:2,:,:])
    return net


def mobilenet_v2(task, progress=True, moco=False, ctonly=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['num_classes'] = 128 # moco_dim
    if 'pretrained' in kwargs:
        moco = False # using pretrained model
        del kwargs['pretrained']
    
    net = MobileNetV2(ctonly=ctonly, moco=moco, **kwargs)
    if moco: # using moco model
        state = torch.load("./chpt/moco_mobilenet.pth")["state_dict"]
        state_dict = {k.replace('encoder_q.',''):state[k] for k in state.keys() if 'encoder_q' in k}

    else: # using pretrained model
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)

    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    
    if task in ['classification']:
        net.classifier_new = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1280, 1),
            )
    
    if ctonly:
        net.features[0][0].weight = torch.nn.Parameter(net.features[0][0].weight[:,:1,:,:])
    else:
        net.features[0][0].weight = torch.nn.Parameter(net.features[0][0].weight[:,:2,:,:])
    return net


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):

    def __init__(self, n_class = 1):
        super().__init__()
        
        self.base_model = resnet18(pretrained=True)
        
        self.base_layers = list(self.base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0[0].weight = torch.nn.Parameter(self.layer0[0].weight[:,:1,:,:]) # only use 1 channel of input
        
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size0 = convrelu(1, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        
        out = self.conv_last(x)        
        
        return out

def ctprocessing(img):
    img = np.asarray(img)
    img = img.astype(np.float32)/255. # normalize value to [0.,1.]

    # adjust brightness
    img = skt.resize(img, (480,480), mode='constant', anti_aliasing=False)
    bandwidth = 255
    img = (img - img.min()) / (img.max() - img.min())
    imhist = (img * bandwidth).astype(np.uint8)
    h = np.histogram(imhist.flatten(), bins=bandwidth+1)
    hmed = ss.medfilt(h[0], kernel_size=51)
    hf = snd.gaussian_filter(hmed, sigma=25)
    hf = np.maximum(0, hf - len(img.flatten())*0.001) # reject 0.1% of mass
    if np.max(hf) > 0:
        hmin = np.nonzero(hf)[0][0] /bandwidth
        hmax = np.nonzero(hf)[0][-1]/bandwidth
        img = (img - hmin) / (hmax - hmin)
    return img


class COVID19DataSet(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train): # from the reference... the size of the image is 480 by 480
        self.imgs = np.reshape(np.array(X_train),[len(X_train),])
        self.labels = np.asarray(y_train)
        assert self.imgs.shape[0] == self.labels.shape[0]

        self.set_indices_train(list(range(len(self))))

    def set_indices_train(self, indices_train):
        self.indices_train = indices_train

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if random.random() < 0.5 and idx in self.indices_train:
            hflip = True
        else: 
            hflip = False
        
        # label = torch.FloatTensor([self.labels[idx]])
        if self.labels[idx] in ['COVID']:
            label = torch.FloatTensor([1.])
        else:
            label = torch.FloatTensor([0.])
        
        img = self.imgs[idx]
        img = Image.fromarray(img)
        img = img.convert("L") # change to greyscale
        
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = ctprocessing(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        return img, label

class COVID19DataSetTest(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test
        self.dims = [480,480]

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x = self.x_test[idx]
        img = Image.fromarray(x).convert('L')
        img = ctprocessing(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img


class LungSegDataSet(torch.utils.data.Dataset):
    def __init__(self, imgs, masks):
        assert imgs.shape[0] == masks.shape[0] # the number of images
        self.imgs = imgs
        self.masks = masks
        self.dims = [480,480]
    
    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):   
        hflip = True if random.random() < 0.5 else False
        img = self.imgs[idx]
        if hflip:
            img = Image.fromarray(img)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = np.asarray(img)
        img = ctprocessing(img)
        img = torch.from_numpy(img).unsqueeze(0).float()

        mask = self.masks[idx]
        if hflip:
            mask = Image.fromarray(mask)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.asarray(mask)
        mask = np.asarray(mask)
        mask = mask.astype(np.float32)/255.
        

        mask = skt.resize(mask, (self.dims[0], self.dims[1]), mode='constant', anti_aliasing=False)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img, mask


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def train_lungseg(epoch, net, trainloader, optimizer, device):
    net.train()  # Set model to training mode
    train_loss = 0.
    epoch_samples = 0
    for batch_idx, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = calc_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        epoch_samples += imgs.size(0)
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')
    return net

def train_classifier(epoch, net, lungseg_net, trainloader, criterion, optimizer, device):
    net.train() # train mode
    train_loss = 0.
    
    for batch_idx, (imgs, labels) in enumerate(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        lungsegs = lungseg_net(imgs)
        optimizer.zero_grad()
        
        logits = net(imgs, lungsegs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('  Training... Epoch: %4d | Iter: %4d/%4d | Train Loss: %.4f'%(epoch, batch_idx+1, len(trainloader), train_loss/(batch_idx+1)), end = '\r')
    print('')

    return net


def validate_classifier(net, testloader, device):
    probs = []
    gts = []
    net.eval() # eval mode

    with torch.no_grad():
        for batch_idx, (imgs, lungsegs, labels) in enumerate(testloader):
            imgs = imgs.to(device)
            lungsegs = lungsegs.to(device)
            logits = net(imgs, lungsegs)
            probs.append(torch.sigmoid(logits))
            gts.append(labels)

    probs = torch.cat(probs, dim=0)
    preds = torch.round(probs).cpu().numpy()
    probs = probs.cpu().numpy()
    gts = torch.cat(gts, dim=0).cpu().numpy()
    auroc = roc_auc_score(gts, probs)
    precision, recall, thresholds = precision_recall_curve(gts, probs)
    aupr = auc(recall, precision)
    f1 = f1_score(gts, preds)

    num_correct = (preds == gts).sum()
    accuracy = num_correct/gts.shape[0]
    print('  AUROC: %5.4f | AUPR: %5.4f | F1_Score: %5.4f | Accuracy: %5.4f (%d/%d)'%(auroc, aupr, f1, accuracy, num_correct, gts.shape[0]))
    
    return auroc, aupr, f1, accuracy

# MAIN FUNCTION
def predict(X_test, model = ""):
    net = mobilenet_v2(task = 'classification', moco = False, ctonly = False)
    lungseg_net = ResNetUNet() # load model

    # state = torch.load("/home/medieason/QSRDC2020/upload/5f237d13e14548571f93276a/model.pth")
    # state = torch.load("model.pth")
    state = torch.load(model+"/model.pth") # for submission
    
    net.load_state_dict(state['classifier'])
    lungseg_net.load_state_dict(state['lungseg'])
    dataset = COVID19DataSetTest(X_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    net.eval()
    y_pred = []
    
    with torch.no_grad():
        for idx, img in enumerate(dataloader):
            org_dim = X_test[idx].shape[:2]
            lungseg = torch.sigmoid(lungseg_net(img)).detach().cpu().numpy().squeeze() * 255.
            lungseg = skt.resize(lungseg, org_dim, mode='constant', anti_aliasing=False)
            lungseg = Image.fromarray(lungseg).convert("L")
            lungseg = np.asarray(lungseg)
            lungseg = lungseg.astype(np.float32)/255.
            lungseg = skt.resize(lungseg, (480, 480), mode='constant', anti_aliasing=False) # resize
            lungseg = torch.from_numpy(lungseg).unsqueeze(0).float()
            logit = net(img, lungseg.unsqueeze(0))
            pred = torch.sigmoid(logit)
            print(pred)
            if pred.item() > 0.5:
                y_pred.append('COVID')
            else:
                y_pred.append('NonCOVID')
    return y_pred

# MAIN FUNCTION
def estimate(X_train, y_train):
    """ 
    estimate function to show the procedure of training the models
    Our training procedure is two-fold: 1) lung segmentation model 2) COVID19 classification
    """
    # 1) training lung segmentation
    """
    please download the following two files and place into the directory where 'Model.py' is located.
    lungseg_imgs.npy: https://drive.google.com/file/d/1GMReqHQuDJIKWoiP10lI_h0rtbRA6zkM/view?usp=sharing
    lungseg_masks.npy: https://drive.google.com/file/d/1TFuxCklBVHO4s9ai4jgsvQPsqb7Jdlox/view?usp=sharing
    """
    maxepoch= 100
    batch_size = 10
    lungseg_imgs = np.load("lungseg_imgs.npy", allow_pickle = True)
    lungseg_masks = np.load("lungseg_masks.npy", allow_pickle = True)
    lungseg_dataset = LungSegDataSet(lungseg_imgs, lungseg_masks)
    
    # device = "cuda"
    device = 'cpu'
    lungseg_net = ResNetUNet(n_class=1).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lungseg_net.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(lungseg_dataset, batch_size=batch_size, shuffle=True, num_workers = 1)
    for epoch in range(maxepoch):
        lungseg_net = train_lungseg(epoch, lungseg_net, trainloader, optimizer, device)
        scheduler.step()

    del lungseg_dataset


    # 2) training classifier for COVID19
    """
    when we trained classifier model for deployment, the lung segmentation image outputs are saved as images, and loader 
    again loads the lung segmentation images to feed into the classifier model. 
    However, here, we cannot do that. Instead, we predict the lung segmentation image online and feed into the model along with CT images.
    Because of this, we cannot allocate larger batch size (For example, for our training, we used batch_size=32 and batch_size_test=64), 
    instead, we set batch_size to 5

    When it comes to splitting the dataset into training and validation (here, 'testset') data sets, 
    we used CT image-patient pair information included in "CT-MetaInfo.xlsx".
    But here, we cannot use this information, thus, we randomly split the dataset.
    """

    lr = 0.001
    batch_size = 5
    batch_size_test = 5
    maxepoch= 100
    
    net = mobilenet_v2(task = 'classification', moco = False, ctonly = False)
    dataset = COVID19DataSet(X_train, y_train)
    ndata = len(dataset)
    ntrain = int(0.8*ndata) # split dataset into training and test data sets
    indices = torch.randperm(ndata).tolist()
    trainset = torch.utils.data.Subset(dataset, indices[:ntrain])
    testset = torch.utils.data.Subset(dataset, indices[ntrain:])
    dataset.set_indices_train(trainset.indices)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)   
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers = 1)

    print('==> Start training ..')   
    lungseg_net.eval() # lung segmentation is set to eval mode
    for epoch in range(maxepoch):
        net = train_classifier(epoch, net, lungseg_net, trainloader, criterion, optimizer, device)
        scheduler.step()
        if epoch%5 == 0:
            validate_classifier(net, testloader, device)

    model_dict = {'classifier': net.to('cpu').state_dict() , 'lungseg': lungseg_net.to('cpu').state_dict()}
    return model_dict