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

from medtk.model.nnModules import ComponentModule
from medtk.model.nd import ConvTransposeNd
from medtk.model.blocks import ConvNormAct


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.dim = dim
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            mode = 'trilinear' if dim == 3 else 'bilinear'
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                # ConvBnRelu(dim, in_channels, out_channels, kernel_size=1, padding=0)
                ConvNormAct(dim, in_channels, out_channels, kernel_size=3, padding=1)  # better
            )
        else:
            self.up = ConvTransposeNd(dim)(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            ConvNormAct(dim, 2 * out_channels, out_channels, kernel_size=3, padding=1),
            ConvNormAct(dim, out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: smaller but deeper feature map
            x2: larger size

        Returns:

        """
        x1 = self.up(x1)
        # print(x1.shape)
        if self.dim == 3:
            # input is CDHW
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
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


class UNetDecoder(ComponentModule):
    def __init__(self,
                 dim: int,
                 in_channels: (list, tuple),
                 out_indices: (list, tuple),
                 bilinear=True):
        super(UNetDecoder, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.dim = dim
        self.bilinear = bilinear

        self.ups = self.init_layers()
        self.init_weights()

    def init_layers(self):
        ups = []
        # for i, planes in enumerate(reversed(in_channels[:-1])):  # in_channels = [64, 128, 256, 512, 1024]
        for i in range(len(self.in_channels) - 1):  # in_channels = [64, 128, 256, 512, 1024]
            ups.append(Up(self.dim, self.in_channels[-i - 1], self.in_channels[-i - 2], self.bilinear))
            # print('u', planes * 2, planes)

        ups = nn.ModuleList(ups)
        return ups

    def init_weights(self, bn_std=0.02):
        pass
        # for m in self.modules():
        #     if isinstance(m, ConvNd(self.dim)):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, BatchNormNd(self.dim)):
        #         nn.init.normal_(m.weight, 1.0, bn_std)
        #         m.bias.data.zero_()

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
        return outs


if __name__ == "__main__":
    import torch

    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    init_seed(666)

    r = UNetDecoder(2, (16, 32, 64, 128), (0, 1, 2, 3))
    print(r)
    r.print_model_params()

    inputs = [
        torch.ones((1, 16, 32, 32)),
        torch.ones((1, 32, 16, 16)),
        torch.ones((1, 64, 8, 8)),
        torch.ones((1, 128, 4, 4)),
    ]

    outs = r(inputs)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))