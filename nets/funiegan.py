# # # # """
# # # #（1）利用多尺度密度块来进行生成器网络增强，深度剖析图像的细节（2）利用残差网络进行生成器的优化，使得上采样保留更多的图像细节（3）增加角度损失对图片增强进行优化，使水下图像整体的对比度得到增强.。
# # # #  > Network architecture of FUnIE-GAN model
# # # #    * Paper: arxiv.org/pdf/1903.09766.pdf
# # # #  > Maintainer: https://github.com/xahidbuffon
# # # # """
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F
# # # #
# # # # class MSRB_Block(nn.Module):
# # # #     def __init__(self, in_channels, out_channels):
# # # #
# # # #         super(MSRB_Block, self).__init__()
# # # #
# # # #         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
# # # #                                   bias=True)
# # # #         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
# # # #                                   bias=True)
# # # #         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
# # # #                                    bias=True)
# # # #         self.relu = nn.LeakyReLU(0.2)
# # # #
# # # #     def forward(self, x):
# # # #             output_3_1 = self.relu(self.conv_3_1(x))
# # # #             output_5_1 = self.relu(self.conv_5_1(x))
# # # #
# # # #             output1 = torch.cat([output_3_1, output_5_1], 1)
# # # #             output2 =  torch.cat([output1, x], 1)
# # # #             output = self.relu(self.confusion(output2))
# # # #             return output
# # # #
# # # # class UNetDown(nn.Module):
# # # #     def __init__(self, in_size, out_size, bn=True):
# # # #         super(UNetDown, self).__init__()
# # # #         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
# # # #         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
# # # #         layers.append(nn.LeakyReLU(0.2))
# # # #         self.model = nn.Sequential(*layers)
# # # #
# # # #     def forward(self, x):
# # # #         return self.model(x)
# # # #
# # # #
# # # # class UNetUp(nn.Module):
# # # #     def __init__(self, in_size, out_size):
# # # #         super(UNetUp, self).__init__()
# # # #         layers = [
# # # #             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
# # # #             nn.BatchNorm2d(out_size, momentum=0.8),
# # # #             nn.ReLU(inplace=True),
# # # #         ]
# # # #         self.model = nn.Sequential(*layers)
# # # #
# # # #     def forward(self, x, skip_input):
# # # #         x = self.model(x)
# # # #         skip_input1 = x + skip_input
# # # #         x = torch.cat((x, skip_input1), 1)
# # # #         return x
# # # #
# # # #
# # # # class GeneratorFunieGAN(nn.Module):
# # # #     """ A 5-layer UNet-based generator as described in the paper
# # # #     """
# # # #     def __init__(self, in_channels=3, out_channels=3):
# # # #         super(GeneratorFunieGAN, self).__init__()
# # # #         # encoding layers
# # # #         self.down1 = UNetDown(in_channels, 32, bn=False)
# # # #         self.Block1 = MSRB_Block(32, 32)
# # # #         self.down2 = UNetDown(32, 128)
# # # #         self.Block2 = MSRB_Block(128, 128)
# # # #         self.down3 = UNetDown(128, 256)
# # # #         self.Block3 = MSRB_Block(256, 256)
# # # #         self.down4 = UNetDown(256, 256)
# # # #         self.Block4 = MSRB_Block(256, 256)
# # # #         self.down5 = UNetDown(256, 256, bn=False)
# # # #         self.Block5 = MSRB_Block(256, 256)
# # # #         # decoding layers
# # # #         self.up1 = UNetUp(256, 256)
# # # #         self.up2 = UNetUp(512, 256)
# # # #         self.up3 = UNetUp(512, 128)
# # # #         self.up4 = UNetUp(256, 32)
# # # #         self.final = nn.Sequential(
# # # #             nn.Upsample(scale_factor=2),
# # # #             nn.ZeroPad2d((1, 0, 1, 0)),
# # # #             nn.Conv2d(64, out_channels, 4, padding=1),
# # # #             nn.Tanh(),
# # # #         )
# # # #
# # # #     def forward(self, x):
# # # #         d1 = self.down1(x)
# # # #         b1 = self.Block1(d1)
# # # #         d2 = self.down2(b1)
# # # #         b2 = self.Block2(d2)
# # # #         d3 = self.down3(b2)
# # # #         b3 = self.Block3(d3)
# # # #         d4 = self.down4(b3)
# # # #         b4 = self.Block4(d4)
# # # #         d5 = self.down5(b4)
# # # #         b5 = self.Block5(d5)
# # # #         u1 = self.up1(b5, d4)
# # # #         u2 = self.up2(u1, d3)
# # # #         u3 = self.up3(u2, d2)
# # # #         u45 = self.up4(u3, d1)
# # # #         return self.final(u45)
# # # #
# # # #
# # # # class DiscriminatorFunieGAN(nn.Module):
# # # #     """ A 4-layer Markovian discriminator as described in the paper
# # # #     """
# # # #     def __init__(self, in_channels=3):
# # # #         super(DiscriminatorFunieGAN, self).__init__()
# # # #
# # # #         def discriminator_block(in_filters, out_filters, bn=True):
# # # #             #Returns downsampling layers of each discriminator block
# # # #             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
# # # #             if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
# # # #             layers.append(nn.LeakyReLU(0.2, inplace=True))
# # # #             return layers
# # # #
# # # #         self.model = nn.Sequential(
# # # #             *discriminator_block(in_channels*2, 32, bn=False),
# # # #             *discriminator_block(32, 64),
# # # #             *discriminator_block(64, 128),
# # # #             *discriminator_block(128, 256),
# # # #             nn.ZeroPad2d((1, 0, 1, 0)),
# # # #             nn.Conv2d(256, 1, 4, padding=1, bias=False)
# # # #         )
# # # #
# # # #     def forward(self, img_A, img_B):
# # # #         # Concatenate image and condition image by channels to produce input
# # # #         img_input = torch.cat((img_A, img_B), 1)
# # # #         return self.model(img_input)
# # # #
# # # """
# # #  > Network architecture of FUnIE-GAN model
# # #    * Paper: arxiv.org/pdf/1903.09766.pdf
# # #  > Maintainer: https://github.com/xahidbuffon
# # # """
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# # class MSRB_Block(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #
# #         super(MSRB_Block, self).__init__()
# #
# #         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
# #                                   bias=True)
# #         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
# #                                   bias=True)
# #         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
# #                                    bias=True)
# #         self.relu = nn.LeakyReLU(0.2)
# #
# #     def forward(self, x):
# #         output_3_1 = self.relu(self.conv_3_1(x))
# #         output_5_1 = self.relu(self.conv_5_1(x))
# #
# #         output1 = torch.cat([output_3_1, output_5_1], 1)
# #         output2 =  torch.cat([output1, x], 1)
# #         output = self.relu(self.confusion(output2))
# #         return output
# #
# # class UNetDown(nn.Module):
# #     def __init__(self, in_size, out_size, bn=True):
# #         super(UNetDown, self).__init__()
# #         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
# #         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
# #         layers.append(nn.LeakyReLU(0.2))
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, x):
# #         return self.model(x)
# #
# #
# # class UNetUp(nn.Module):
# #     def __init__(self, in_size, out_size):
# #         super(UNetUp, self).__init__()
# #         layers = [
# #             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(out_size, momentum=0.8),
# #             nn.ReLU(inplace=True),
# #         ]
# #         self.model = nn.Sequential(*layers)
# #
# #     def forward(self, x, skip_input):
# #         x = self.model(x)
# #         x = torch.cat((x, skip_input), 1)
# #         return x
# #
# #
# # class GeneratorFunieGAN(nn.Module):
# #     """ A 5-layer UNet-based generator as described in the paper
# #     """
# #     def __init__(self, in_channels=3, out_channels=3):
# #         super(GeneratorFunieGAN, self).__init__()
# #         # encoding layers
# #         self.down1 = UNetDown(in_channels, 32, bn=False)
# #         self.Block1 = MSRB_Block(32, 32)
# #         self.down2 = UNetDown(32, 128)
# #         self.Block2 = MSRB_Block(128, 128)
# #         self.down3 = UNetDown(128, 256)
# #         self.Block3 = MSRB_Block(256, 256)
# #         self.down4 = UNetDown(256, 256)
# #         self.Block4 = MSRB_Block(256, 256)
# #         self.down5 = UNetDown(256, 256, bn=False)
# #         self.Block5 = MSRB_Block(256, 256)
# #         # decoding layers
# #         self.up1 = UNetUp(256, 256)
# #         self.up2 = UNetUp(512, 256)
# #         self.up3 = UNetUp(512, 128)
# #         self.up4 = UNetUp(256, 32)
# #         self.final = nn.Sequential(
# #             nn.Upsample(scale_factor=2),
# #             nn.ZeroPad2d((1, 0, 1, 0)),
# #             nn.Conv2d(64, out_channels, 4, padding=1),
# #             nn.Tanh(),
# #         )
# #
# #     def forward(self, x):
# #         d1 = self.down1(x)
# #         b1 = self.Block1(d1)
# #         d2 = self.down2(b1)
# #         b2 = self.Block2(d2)
# #         d3 = self.down3(b2)
# #         b3 = self.Block3(d3)
# #         d4 = self.down4(b3)
# #         b4 = self.Block4(d4)
# #         d5 = self.down5(b4)
# #         b5 = self.Block5(d5)
# #         u1 = self.up1(b5, d4)
# #         u2 = self.up2(u1, d3)
# #         u3 = self.up3(u2, d2)
# #         u45 = self.up4(u3, d1)
# #         return self.final(u45)
# #
# #
# # class DiscriminatorFunieGAN(nn.Module):
# #     """ A 4-layer Markovian discriminator as described in the paper
# #     """
# #     def __init__(self, in_channels=3):
# #         super(DiscriminatorFunieGAN, self).__init__()
# #
# #         def discriminator_block(in_filters, out_filters, bn=True):
# #             #Returns downsampling layers of each discriminator block
# #             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
# #             if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
# #             layers.append(nn.LeakyReLU(0.2, inplace=True))
# #             return layers
# #
# #         self.model = nn.Sequential(
# #             *discriminator_block(in_channels*2, 32, bn=False),
# #             *discriminator_block(32, 64),
# #             *discriminator_block(64, 128),
# #             *discriminator_block(128, 256),
# #             nn.ZeroPad2d((1, 0, 1, 0)),
# #             nn.Conv2d(256, 1, 4, padding=1, bias=False)
# #         )
# #
# #     def forward(self, img_A, img_B):
# #         # Concatenate image and condition image by channels to produce input
# #         img_input = torch.cat((img_A, img_B), 1)
# #         return self.model(img_input)
# #
# # -*- condeing = utf-8 -*-
# # @Time : 2021/9/7 16:05
# # Author : 磊
# # @File : funiegan2.py
# # @Software : PyCharm
# """
#  > Network architecture of FUnIE-GAN model
#    * Paper: arxiv.org/pdf/1903.09766.pdf
#  > Maintainer: https://github.com/xahidbuffon
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MSRB_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(MSRB_Block, self).__init__()
#
#         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                                   bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
#                                   bias=True)
#         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                                    bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#
#         output1 = torch.cat([output_3_1, output_5_1], 1)
#         output2 =  torch.cat([output1, x], 1)
#         output = self.relu(self.confusion(output2))
#         return output
#
# class MSRB1_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(MSRB1_Block, self).__init__()
#
#         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                                   bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
#                                   bias=True)
#         self.conv_7_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3,
#                                   bias=True)
#         self.confusion = nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                                    bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#         output_7_1 = self.relu(self.conv_7_1(x))
#         output1 = torch.cat([output_3_1, output_5_1], 1)
#         output2 = torch.cat([output1, output_7_1], 1)
#         output3 = torch.cat([output2, x], 1)
#         output = self.relu(self.confusion(output3))
#         return output
#
# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, bn=True):
#         super(UNetDown, self).__init__()
#         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
#         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
#         layers.append(nn.LeakyReLU(0.2))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, bn=True):
#         super(UNetDown, self).__init__()
#         layers1 = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False)]
#         layers2 = [nn.Conv2d(in_size, out_size, 5, 2, 2, bias=False)]
#         layers1.append(nn.LeakyReLU(0.2))
#         layers2.append(nn.LeakyReLU(0.2))
#         layers = torch.cat(layers1, layers2)
#         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
#
# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(UNetUp, self).__init__()
#         layers = [
#             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(out_size, momentum=0.8),
#             nn.ReLU(inplace=True),
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, skip_input):
#         x = self.model(x)
#         skip_input = skip_input + x
#         x = torch.cat((x, skip_input), 1)
#         return x
#
#
# class GeneratorFunieGAN(nn.Module):
#     """ A 5-layer UNet-based generator as described in the paper
#     """
#     def __init__(self, in_channels=3, out_channels=3):
#         super(GeneratorFunieGAN, self).__init__()
#         # encoding layers
#         self.down1 = UNetDown(in_channels, 32, bn=False)
#
#         self.down2 = UNetDown(32, 128)
#         self.down3 = UNetDown(128, 256)
#         self.down4 = UNetDown(256, 256)
#         self.Block1 = MSRB1_Block(256, 256)
#         # decoding layers
#         self.up1 = UNetUp(256, 256)
#         self.Block2 = MSRB_Block(512, 512)
#         self.up2 = UNetUp(512, 128)
#         self.Block3 = MSRB_Block(256, 256)
#         self.up3 = UNetUp(256, 32)
#         self.Block4 = MSRB_Block(64, 64)
#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(64, out_channels, 4, padding=1),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         b1 = self.Block1(d4)
#         u1 = self.up1(b1, d3)
#         b2 = self.Block2(u1)
#         u2 = self.up2(b2, d2)
#         b3 = self.Block3(u2)
#         u3 = self.up3(b3, d1)
#         b4 = self.Block4(u3)
#         return self.final(b4)
#
#
# class DiscriminatorFunieGAN(nn.Module):
#     """ A 4-layer Markovian discriminator as described in the paper
#     """
#     def __init__(self, in_channels=3):
#         super(DiscriminatorFunieGAN, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             #Returns downsampling layers of each discriminator block
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels*2, 32, bn=False),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(256, 1, 4, padding=1, bias=False)
#         )
#
#     def forward(self, img_A, img_B):
#         # Concatenate image and condition image by channels to produce input
#         img_input = torch.cat((img_A, img_B), 1)
#         return self.model(img_input)
#
#
#####两种算法组合

# import torch
# import torch.nn as nn
# from torch import Tensor
# import torch.nn.functional as F
# from torchvision.models.mobilenetv2 import _make_divisible
#
# class MSRB_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(MSRB_Block, self).__init__()
#
#         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                                   bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
#                                   bias=True)
#         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                                    bias=True)
#         self.relu = nn.LeakyReLU(0.2)
#
#     def forward(self, x):
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#
#         output1 = torch.cat([output_3_1, output_5_1], 1)
#         output2 =  torch.cat([output1, x], 1)
#         output = self.relu(self.confusion(output2))
#         return output
#
#
# class SqueezeExcitation(nn.Module):
#     def __init__(self, input_c: int, squeeze_factor: int = 4):
#         super(SqueezeExcitation, self).__init__()
#         squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
#         self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
#         self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
#
#     def forward(self, x: Tensor) -> Tensor:
#         scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
#         scale = self.fc1(scale)
#         scale = F.relu(scale, inplace=True)
#         scale = self.fc2(scale)
#         scale = F.hardsigmoid(scale, inplace=True)
#         return scale * x
#
# class MSRB_Block1(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(MSRB_Block1, self).__init__()
#
#         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                                   bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
#                                   bias=True)
#         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                                    bias=True)
#         self.SE = SqueezeExcitation(out_channels * 2)
#         self.relu = nn.LeakyReLU(0.2)
#
#     def forward(self, x):
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#         # output0 = SqueezeExcitation(x)
#         # output1 = torch.cat([output_3_1, output_5_1, output0], 1)
#         output1 = torch.cat([output_3_1, output_5_1], 1)
#         output0 = self.SE(output1)
#         #output2 =  torch.cat([output0, x], 1)
#         output2 = self.confusion(output0)
#         output = torch.add(output2, x)
#         return output
# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, bn=True):
#         super(UNetDown, self).__init__()
#         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
#         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
#         layers.append(nn.LeakyReLU(0.2))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
# # class UNetDown(nn.Module):
# #     def __init__(self, in_size, out_size, bn=True):
# #         super(UNetDown, self).__init__()
# #         layers1 = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False)]
# #         layers2 = [nn.Conv2d(in_size, out_size, 5, 2, 2, bias=False)]
# #         layers1.append(nn.LeakyReLU(0.2))
# #         layers2.append(nn.LeakyReLU(0.2))
# #         layers = torch.cat(layers1, layers2)
# #         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
# #         self.model = nn.Sequential(*layers1,*layers2)
#
#     # def forward(self, x):
#     #     return self.model(x)
#
#
# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(UNetUp, self).__init__()
#         layers = [
#             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(out_size, momentum=0.8),
#             nn.ReLU(inplace=True),
#             # SqueezeExcitation(out_size)
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, skip_input):
#         x = self.model(x)
# #       skip_input = skip_input + x
#         x = torch.cat((x, skip_input), 1)
#         return x
#
#
# class GeneratorFunieGAN(nn.Module):
#     """ A 5-layer UNet-based generator as described in the paper
#     """
#     def __init__(self, in_channels=3, out_channels=3):
#         super(GeneratorFunieGAN, self).__init__()
#         # encoding layers
#         self.down1 = UNetDown(in_channels, 32, bn=False)
#         self.Block1 = MSRB_Block(32, 32)
#         self.down2 = UNetDown(32, 128)
#         self.Block2 = MSRB_Block(128, 128)
#         self.down3 = UNetDown(128, 256)
#         self.Block3 = MSRB_Block(256, 256)
#         self.down4 = UNetDown(256, 256)
#         self.Block4 = MSRB_Block1(256, 256)
#         # self.down5 = UNetDown(256, 256, bn=False)
#         # self.Block5 = MSRB_Block(256, 128)
#         # decoding layers
#         # self.up1 = UNetUp(256, 256)
#         self.up2 = UNetUp(256, 256)
#         self.up3 = UNetUp(512, 128)
#         self.up4 = UNetUp(256, 32)
#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(64, out_channels, 4, padding=1),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         d1 = self.down1(x)
#         b1 = self.Block1(d1)
#         d2 = self.down2(b1)
#         b2 = self.Block2(d2)
#         d3 = self.down3(b2)
#         b3 = self.Block3(d3)
#         d4 = self.down4(b3)
#         b4 = self.Block4(d4)
#         # d5 = self.down5(b4)
#         # b5 = self.Block5(d5)
#         # u1 = self.up1(b5, d4)
#         u2 = self.up2(b4, d3)
#         u3 = self.up3(u2, d2)
#         u45 = self.up4(u3, d1)
#         return self.final(u45)
#
#
# class DiscriminatorFunieGAN(nn.Module):
#     """ A 4-layer Markovian discriminator as described in the paper
#     """
#     def __init__(self, in_channels=3):
#         super(DiscriminatorFunieGAN, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             #Returns downsampling layers of each discriminator block
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels*2, 32, bn=False),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(256, 1, 4, padding=1, bias=False)
#         )
#
#     def forward(self, img_A, img_B):
#         # Concatenate image and condition image by channels to produce input
#         img_input = torch.cat((img_A, img_B), 1)
#         return self.model(img_input)
#
#
"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MSRB_Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#
#         super(MSRB_Block, self).__init__()
#
#         self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                                   bias=True)
#         self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
#                                   bias=True)
#         self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                                    bias=True)
#         self.relu = nn.LeakyReLU(0.2)
#
#     def forward(self, x):
#         output_3_1 = self.relu(self.conv_3_1(x))
#         output_5_1 = self.relu(self.conv_5_1(x))
#
#         output1 = torch.cat([output_3_1, output_5_1], 1)
#         output2 =  torch.cat([output1, x], 1)
#         output = self.relu(self.confusion(output2))
#         return output
#
# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, bn=True):
#         super(UNetDown, self).__init__()
#         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
#         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
#         layers.append(nn.LeakyReLU(0.2))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(UNetUp, self).__init__()
#         layers = [
#             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(out_size, momentum=0.8),
#             nn.ReLU(inplace=True),
#         ]
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x, skip_input):
#         x = self.model(x)
#         x = torch.cat((x, skip_input), 1)
#         return x
#
#
# class GeneratorFunieGAN(nn.Module):
#     """ A 5-layer UNet-based generator as described in the paper
#     """
#     def __init__(self, in_channels=3, out_channels=3):
#         super(GeneratorFunieGAN, self).__init__()
#         # encoding layers
#         self.down1 = UNetDown(in_channels, 32, bn=False)
#         self.Block1 = MSRB_Block(32, 32)
#         self.down2 = UNetDown(32, 128)
#         self.Block2 = MSRB_Block(128, 128)
#         self.down3 = UNetDown(128, 256)
#         self.Block3 = MSRB_Block(256, 256)
#         self.down4 = UNetDown(256, 256)
#         self.Block4 = MSRB_Block(256, 256)
#         self.down5 = UNetDown(256, 256, bn=False)
#         self.Block5 = MSRB_Block(256, 256)
#         # decoding layers
#         self.up1 = UNetUp(256, 256)
#         self.up2 = UNetUp(512, 256)
#         self.up3 = UNetUp(512, 128)
#         self.up4 = UNetUp(256, 32)
#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(64, out_channels, 4, padding=1),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         d1 = self.down1(x)
#         b1 = self.Block1(d1)
#         d2 = self.down2(b1)
#         b2 = self.Block2(d2)
#         d3 = self.down3(b2)
#         b3 = self.Block3(d3)
#         d4 = self.down4(b3)
#         b4 = self.Block4(d4)
#         d5 = self.down5(b4)
#         b5 = self.Block5(d5)
#         u1 = self.up1(b5, d4)
#         u2 = self.up2(u1, d3)
#         u3 = self.up3(u2, d2)
#         u45 = self.up4(u3, d1)
#         return self.final(u45)
#
#
# class DiscriminatorFunieGAN(nn.Module):
#     """ A 4-layer Markovian discriminator as described in the paper
#     """
#     def __init__(self, in_channels=3):
#         super(DiscriminatorFunieGAN, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, bn=True):
#             #Returns downsampling layers of each discriminator block
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels*2, 32, bn=False),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(256, 1, 4, padding=1, bias=False)
#         )
#
#     def forward(self, img_A, img_B):
#         # Concatenate image and condition image by channels to produce input
#         img_input = torch.cat((img_A, img_B), 1)
#         return self.model(img_input)
#
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import _make_divisible

class MSRB_Block(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(MSRB_Block, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        output1 = torch.cat([output_3_1, output_5_1], 1)
        output2 = torch.cat([output1, x], 1)
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

class MSRB_Block1(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(MSRB_Block1, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_7_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.SE = SqueezeExcitation(out_channels * 3)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        output_7_1 = self.relu(self.conv_7_1(x))
        # output0 = SqueezeExcitation(x)
        # output1 = torch.cat([output_3_1, output_5_1, output0], 1)
        output1 = torch.cat([output_3_1, output_5_1, output_7_1], 1)
        output0 = self.SE(output1)
        #output2 =  torch.cat([output0, x], 1)
        output2 = self.confusion(output0)
        output = torch.add(output2, x)
        return output

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, bn=True):
#         super(UNetDown, self).__init__()
#         layers1 = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False)]
#         layers2 = [nn.Conv2d(in_size, out_size, 5, 2, 2, bias=False)]
#         layers1.append(nn.LeakyReLU(0.2))
#         layers2.append(nn.LeakyReLU(0.2))
#         layers = torch.cat(layers1, layers2)
#         if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
#         self.model = nn.Sequential(*layers1,*layers2)

    # def forward(self, x):
    #     return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_size)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
#       skip_input = skip_input + x
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        # self.Block1 = MSRB_Block(32, 32)
        self.down2 = UNetDown(32, 128)
        # self.Block2 = MSRB_Block(128, 128)
        self.down3 = UNetDown(128, 256)
        # self.Block3 = MSRB_Block(256, 256)
        self.down4 = UNetDown(256, 256)
        self.Block4 = MSRB_Block1(256, 256)
        # self.down5 = UNetDown(256, 256, bn=False)
        # self.Block5 = MSRB_Block(256, 128)
        # decoding layers
        # self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(256, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        # b1 = self.Block1(d1)
        d2 = self.down2(d1)
        # b2 = self.Block2(d2)
        d3 = self.down3(d2)
        # b3 = self.Block3(d3)
        d4 = self.down4(d3)
        b4 = self.Block4(d4)
        # d5 = self.down5(b4)
        # b5 = self.Block5(d5)
        # u1 = self.up1(b5, d4)
        u2 = self.up2(b4, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


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

