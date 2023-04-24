import torch
import torch.nn as nn
import torch.nn.functional as F
from SPP import ASPP


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
    def __init__(self, nInputChannels, os, backbone_type):
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
        layer3_in = 512
        layer3_out = 48

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, aspp_rates)


        self.conv2 = nn.Conv2d(lowInputChannels, lowOutputChannels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(lowOutputChannels)

        self.conv3 = nn.Conv2d(layer3_in, layer3_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(layer3_out)

        self.cat_conv = nn.Sequential(
            nn.Conv2d(asppOutputChannels + layer3_out, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

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
        layer1_features = x

        x = self.layer2(x)  # 512,44,44
        layer2_features = x

        x = self.layer3(x)  # 1024,22,22
        x = self.layer4(x)  #2048,22,22
        x_aspp = self.aspp(x)    # 256,22,22

        layer2_features = self.conv3(layer2_features)
        layer2_features = self.bn3(layer2_features)

        x_layer2 = F.upsample(x_aspp, layer2_features.size()[2:], mode='bilinear', align_corners=True)
        x_cat = torch.cat((layer2_features, x_layer2), dim=1)
        x_cat = self.cat_conv(x_cat)

        layer1_features = self.conv2(layer1_features)
        layer1_features = self.bn2(layer1_features)


        return layer1_features, layer2_features, x_cat


class CoattentionModel(nn.Module):
    def __init__(self, num_classes, all_channel=256, all_dim=56 * 56):
        super(CoattentionModel, self).__init__()
        self.encoder = resnet50_aspp(3, 16, 'resnet50')

        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim

        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)

        self.last_conv1 = nn.Sequential(
            nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )
        self.last_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input1, input2, gt1):

        input_size = input1.size()[2:]
        exemplar_layer1, exemplar_layer2, exemplar = self.encoder(input1)
        query_layer1, query_layer2, query = self.encoder(input2)

        GT_DOWN = gt1
        GT_DOWN = F.interpolate(GT_DOWN, scale_factor=0.5)
        GT_DOWN = F.interpolate(GT_DOWN, scale_factor=0.5)
        GT_DOWN = F.interpolate(GT_DOWN, scale_factor=0.5)
        GT_flat = GT_DOWN.view(GT_DOWN.size()[0], GT_DOWN.size()[1], -1).contiguous()  # N,C,H*W

        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]

        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W  4,256,3136
        query_flat = query.view(-1, query.size()[1], all_dim) # 4,256,3136
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num 4,3136,256
        exemplar_corr = self.linear_e(exemplar_t)  # 4,3136,256

        A = torch.bmm(exemplar_corr, query_flat)  # 4,3136,3136
        A1 = F.softmax(A.clone(), dim=1)

        exemplar_flat = exemplar_flat * GT_flat
        query_att = torch.bmm(exemplar_flat, A1).contiguous()

        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_mask = self.gate(input2_att)
        input2_mask = self.gate_s(input2_mask)
        input2_att = input2_att * input2_mask
        input2_att = torch.cat([input2_att, query], 1)
        input2_att = self.conv2(input2_att)
        input2_att = self.bn2(input2_att)
        input2_att = self.prelu(input2_att)
        input2_att = F.upsample(input2_att, query_layer1.size()[2:], mode='bilinear')
        input2_att = torch.cat((input2_att, query_layer1), dim=1)
        x2 = self.last_conv1(input2_att)
        x2_main = self.last_conv2(input2_att)
        x2 = F.upsample(x2, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        x2_main = F.upsample(x2_main, input_size, mode='bilinear')

        return x2, x2_main         # shape: NxCx

