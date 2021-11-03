#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from medtk.model.backbones.unet_enc import Down
from medtk.model.nnModules import ComponentModule
from medtk.model.nd import ConvNd, BatchNormNd, ConvTransposeNd


class DoubleConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvNd(dim)(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            BatchNormNd(dim)(out_channels),
            nn.ReLU(inplace=True),
            ConvNd(dim)(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            BatchNormNd(dim)(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.dim = dim
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = ConvTransposeNd(dim)(in_channels, in_channels, kernel_size=2, stride=2)

        self.squeeze = ConvNd(dim)(in_channels, in_channels // 2, kernel_size=1)
        self.conv = DoubleConv(dim, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape)
        x1 = self.squeeze(x1)
        # print(x1.shape)
        if self.dim == 3:
            # input is CDHW
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
            # print(diffX, diffY, diffZ)
        elif self.dim == 2:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        else:
            raise NotImplementedError
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class IterUNetDecoder(ComponentModule):
    def __init__(self, dim, in_channels, out_indices, bilinear=True):
        super(IterUNetDecoder, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.dim = dim
        self.bilinear = bilinear

        ups = []
        for i, planes in enumerate(reversed(in_channels[:-1])):
            # in_channels = [64, 128, 256, 512, 1024]
            ups.append(Up(dim, planes * 2, planes, bilinear))
            # print('u', planes * 2, planes)

        self.ups = nn.ModuleList(ups)

        self.iterations = 2
        # define the network MiniUNet layers
        self.model_miniunet = nn.ModuleList([MiniUNet(dim, in_channels[0] * 2, in_channels[0], bilinear)] * self.iterations)

    def forward(self, x):
        assert len(self.in_channels) == len(x)
        outs_up = [x[-1]]
        for i in range(len(self.in_channels) - 1):
            in1, in2 = outs_up[-1], x[-i - 2]
            # print(i, in1.shape, in2.shape)
            y = self.ups[i](in1, in2)
            # print(y.shape)
            outs_up.append(y)

        outs_up.reverse()

        outs = []
        for i in self.out_indices:
            outs.append(outs_up[i])

        x1, x2 = x[0], outs_up[0]
        # print(x1.shape, x2.shape)
        for i in range(self.iterations):
            x = torch.cat([x1, x2], dim=1)
            # print(x.shape)
            _, x2 = self.model_miniunet[i](x)
            # print(_.shape, x2.shape)
        return [x2]


class MiniUNet(ComponentModule):
    def __init__(self, dim, in_channels, out_channels, bilinear):
        super(MiniUNet, self).__init__()
        self.dim = dim
        self.in_channels = in_channels

        self.inc = DoubleConv(dim, in_channels, out_channels)
        self.down1 = Down(dim, out_channels, out_channels*2)
        self.down2 = Down(dim, out_channels*2, out_channels*4)
        self.down3 = Down(dim, out_channels*4, out_channels*8)
        self.up1 = Up(dim, out_channels*8, out_channels*4, bilinear)
        self.up2 = Up(dim, out_channels*4, out_channels*2, bilinear)
        self.up3 = Up(dim, out_channels*2, out_channels, bilinear)
        # self.outc = ConvNd(self.dim)(out_channels, n_classes, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # logits = self.outc(x)
        # return x1, x, logits
        return x1, x


if __name__ == '__main__':
    # m = MiniUNet(3, 2, 16, 32)
    # data = torch.rand(1, 16, 32, 32, 32)
    # ps = m(data)
    # [print(p.shape) for p in ps]

    data = [torch.rand(1, 16, 32, 32, 32), torch.rand(1, 32, 16, 16, 16)]
    d = IterUNetDecoder(3, [16, 32], [0])
    outs = d(data)
    [print(o.shape) for o in outs]