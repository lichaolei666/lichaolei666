- 👋 Hi, I’m @lichaolei666
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...# -*- condeing = utf-8 -*-
# @Time : 2021/9/11 10:32
# Author : 磊
# @File : funiegan4.py
# @Software : PyCharm

"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import _make_divisible
from torch import nn, Tensor

class MSRB_Block(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(MSRB_Block, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        output1 = torch.cat([output_3_1, output_5_1], 1)
        output2 =  torch.cat([output1, x], 1)
        output = self.relu(self.confusion(output2))
        return output

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),nn.Conv2d(in_size, out_size, 5, 2, 2, bias=False)]
        layers[0].append(nn.LeakyReLU(0.2))
        layers[1].append(nn.LeakyReLU(0.2))
        layers1 = torch.cat(layers[0],layers[1])
        if bn: layers1.append(nn.BatchNorm2d(out_size, momentum=0.8))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        skip_input = skip_input + x
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)

        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        self.Block1 = MSRB_Block(256, 256)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.Block2 = MSRB_Block(512, 512)
        self.up2 = UNetUp(512, 256)
        self.Block3 = MSRB_Block(512, 512)
        self.up3 = UNetUp(512, 128)
        self.Block4 = MSRB_Block(256, 256)
        self.up4 = UNetUp(256, 32)
        self.Block5 = MSRB_Block(64, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        b1 = self.Block1(d5)
        u1 = self.up1(b1, d4)
        b2 = self.Block2(u1)
        u2 = self.up2(b2, d3)
        b3 = self.Block3(u2)
        u3 = self.up3(b3, d2)
        b4 = self.Block4(u3)
        u45 = self.up4(b4, d1)
        b5 = self.Block5(u45)
        return self.final(b5)


class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


- 📫 How to reach me ...

<!---
lichaolei666/lichaolei666 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
