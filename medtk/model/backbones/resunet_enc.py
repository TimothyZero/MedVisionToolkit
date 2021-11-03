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

from medtk.model.nnModules import ComponentModule
from medtk.model.nd import ConvNd, BatchNormNd, MaxPoolNd
from medtk.model.blocks import ConvNormAct


class DoubleConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvNormAct(dim, in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ConvNormAct(dim, out_channels, out_channels, kernel_size=kernel_size, padding=padding, do_act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.double_conv(x)
        return self.act(out + x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, dim, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ConvNormAct(dim, in_channels, out_channels, kernel_size=2, padding=2),
            DoubleConv(dim, out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ResUNetEncoder(ComponentModule):
    def __init__(self, dim, in_channels, base_width=16, stages=4, out_indices=(0, 1, 2, 3, 4)):
        super(ResUNetEncoder, self).__init__()
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.dim = dim
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.base_width = base_width
        self.stages = stages

        self.inc = DoubleConv(dim, in_channels, base_width)

        downs = []
        for i in range(stages):
            # base_width = 64, i = 0-3, 64 -> 128 -> 256 -> 512
            planes = base_width * pow(2, i)
            downs.append(Down(dim, planes, planes * 2))
            # print('d', planes, planes * 2)

        self.downs = nn.ModuleList(downs)

    def forward(self, x):
        # x torch.Size([1, 3, 32, 32])
        x = self.inc(x)  # x1 torch.Size([1, 64, 32, 32])

        outs_down = [x]
        for i in range(self.stages):
            x = self.downs[i](x)
            # print(x.shape)
            outs_down.append(x)

        outs = []
        for i in self.out_indices:
            outs.append(outs_down[i])
        return outs
