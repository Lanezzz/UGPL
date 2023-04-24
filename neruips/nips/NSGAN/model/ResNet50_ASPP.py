import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.SPP import ASPP_simple, ASPP
from model.ResNet import ResNet101, ResNet18, ResNet34, ResNet50

import time

INPUT_SIZE = 512
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

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

class resnet50_aspp(nn.Module):
    def __init__(self, nInputChannels, n_classes, os, backbone_type):
        super(resnet50_aspp, self).__init__()
        self.inplanes = 64

        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            aspp_rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            aspp_rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
        elif os == 32:
            strides = [1, 2, 2, 2]
            rates = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        assert backbone_type == 'resnet50'

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 6, 3]

        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], rate=rates[3])


        asppInputChannels = 2048
        asppOutputChannels = 256
        lowInputChannels = 256
        lowOutputChannels = 48

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)

        self.last_conv = nn.Sequential(
            nn.Conv2d(asppOutputChannels + lowOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        )

        self.conv2 = nn.Conv2d(lowInputChannels, lowOutputChannels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(lowOutputChannels)

    def _make_layer(self, planes, blocks, stride=1, rate=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img):

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)   # 256,88,88

        low_level_features = x

        x = self.layer2(x)  # 512,44,44

        x = self.layer3(x)  # 1024,22,22

        x = self.layer4(x)  #2048,22,22

        x = self.aspp(x)    # 256,22,22

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = F.upsample(x, low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat((x, low_level_features), dim=1)

        x_final = self.last_conv(x_cat)
        x_final = F.upsample(x_final, img.size()[2:], mode='bilinear', align_corners=True)

        return x_cat, x_final

