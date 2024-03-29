# Adapted from https://github.com/nizhenliang/RAUNet/blob/master/RAUNet.py

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class AAM(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(AAM, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight=self.conv3(weight_low+weight_high)
        low = self.conv4(input_low)
        return input_high+low.mul(weight)

class RSAUdiff(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, pretrained=None):
        super().__init__()
        assert num_channels == 3
        self.w = 256
        self.h = 256
        self.num_classes = num_classes
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.gau3 = AAM(filters[2], filters[2])
        self.gau2 = AAM(filters[1], filters[1])
        self.gau1 = AAM(filters[0], filters[0])


        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 128, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(128, 128, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(128, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x1, x2):
        # Encoder
        # for x2,siamese share weights
        x2 = self.firstconv(x2)
        x2 = self.firstbn(x2)
        x2 = self.firstrelu(x2)
        x2 = self.firstmaxpool(x2)
        e21 = self.encoder1(x2)
        e22 = self.encoder2(e21)
        e23 = self.encoder3(e22)
        #e24 = self.encoder4(e23)
        
        # for x1,siamese share weights
        x1 = self.firstconv(x1)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e11 = self.encoder1(x1)
        e12 = self.encoder2(e11)
        e13 = self.encoder3(e12)
        e14 = self.encoder4(e13)
        
        # decoder
        d4 = self.decoder4(e14)
        b4 = self.gau3(d4, torch.abs(e13 - e23))
        d3 = self.decoder3(b4)
        b3 = self.gau2(d3, torch.abs(e12 - e22))
        d2 = self.decoder2(b3)
        b2 = self.gau1(d2, torch.abs(e11 - e21))
        d1 = self.decoder1(b2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        #return f5

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out
        
class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x