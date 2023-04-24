import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse

from model.ResNet50_ASPP import resnet50_aspp
parser = argparse.ArgumentParser()

global_step = 0
args = parser.parse_args()


class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.down_rgb_channel = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.down_opt_channel = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.spa_att1 =  nn.Conv2d(256, 1, 1, bias=True)
        self.spa_att2 =  nn.Conv2d(256, 1, 1, bias=True)
        self.spa_att3 =  nn.Conv2d(256, 1, 1, bias=True)
        self.spa_att4 =  nn.Conv2d(256, 1, 1, bias=True)


        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1)
        )

        self.weight_init()


    def forward(self, x_RGB,x_OPT):
        x_RGB = self.down_rgb_channel(x_RGB)
        x_OPT = self.down_opt_channel(x_OPT)

        rgb = self.channel_attention_rgb(self.squeeze_rgb(x_RGB))
        rgb_CA = x_RGB * rgb.expand_as(x_RGB)
        rgb_SA = self.spa_att1(rgb_CA)
        rgb_CA_SA = rgb_CA * rgb_SA

        opt = self.channel_attention_depth(self.squeeze_depth(x_OPT))
        opt_CA = x_OPT * opt.expand_as(x_OPT)
        opt_SA = self.spa_att2(opt_CA)
        opt_CA_SA = opt_CA * opt_SA

        norm_att = torch.softmax(rgb + opt,dim=1)

        norm_RGB = x_RGB * norm_att.expand_as(x_RGB)
        norm_RGB_SA = self.spa_att3(norm_RGB)
        norm_RGB_SA = norm_RGB * norm_RGB_SA

        norm_OPT = x_OPT * norm_att.expand_as(x_OPT)
        norm_OPT_SA = self.spa_att4(norm_OPT)
        norm_OPT_SA = norm_OPT * norm_OPT_SA

        final_rgb = rgb_CA_SA + norm_RGB_SA
        final_opt = opt_CA_SA + norm_OPT_SA

        fuse_feature = torch.cat([final_rgb,final_opt],dim=1)
        final_feature = self.last_conv(fuse_feature)

        return fuse_feature, final_feature

    def weight_init(self):
        print("正在初始化 部分网络参数")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):  # 初始化BN层
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):  # 初始化有序容器，因为容器里有很多不同的层，比如卷积层，线性层之类的，所以继续针对容器里的层初始化
                for n in m.named_children():
                    if isinstance(n, nn.Conv2d):  # 初始化卷积层
                        nn.init.kaiming_normal_(n.weight, mode='fan_in', nonlinearity='relu')
                        if n.bias is not None:
                            nn.init.zeros_(n.bias)
                    elif isinstance(n, (nn.BatchNorm2d, nn.InstanceNorm2d)):  # 初始化BN层
                        nn.init.ones_(n.weight)
                        if n.bias is not None:
                            nn.init.zeros_(n.bias)


class SSL_Model(nn.Module):
    def __init__(self, num_classes, output_stride, img_backbone_type, backbone_type):
        super(SSL_Model, self).__init__()

        self.app_branch = resnet50_aspp(nInputChannels=3, n_classes=num_classes, os=output_stride,
                      backbone_type=img_backbone_type)

        self.mot_branch = resnet50_aspp(nInputChannels=3, n_classes=num_classes, os=output_stride,
                                           backbone_type=backbone_type)


        self.fusion = fusion()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)

        self.weight_init()


    def forward(self, img, flow, is_static=False):

            app_cat, app_out = self.app_branch(img)
            if is_static == True:
                app_out = F.upsample(app_out, img.size()[2:], mode='bilinear', align_corners=True)
                return app_out
            mot_cat, mot_out = self.mot_branch(flow)
            fuse_feature, fuse_out = self.fusion(app_cat, mot_cat)
            """
            # 图的部分
            fuse_feat = fuse_feature.detach()
            confidence_map = self.confidence_conv(fuse_feat)
            # 值的部分
            quality = self.quality_conv(fuse_feat)
            quality = self.avg_pool(quality)
            quality_out = quality.view(quality.size(0), -1)
            quality_mea = self.fc(quality_out)
            """
            fuse_out = F.upsample(fuse_out, img.size()[2:], mode='bilinear', align_corners=True)
            #confidence_map = F.upsample(confidence_map, img.size()[2:], mode='bilinear', align_corners=True)
            return app_out, mot_out, fuse_out

    def weight_init(self):
        print("正在初始化 部分网络参数")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):  # 初始化BN层
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):  # 初始化有序容器，因为容器里有很多不同的层，比如卷积层，线性层之类的，所以继续针对容器里的层初始化
                for n in m.named_children():
                    if isinstance(n, nn.Conv2d):  # 初始化卷积层
                        nn.init.kaiming_normal_(n.weight, mode='fan_in', nonlinearity='relu')
                        if n.bias is not None:
                            nn.init.zeros_(n.bias)
                    elif isinstance(n, (nn.BatchNorm2d, nn.InstanceNorm2d)):  # 初始化BN层
                        nn.init.ones_(n.weight)
                        if n.bias is not None:
                            nn.init.zeros_(n.bias)