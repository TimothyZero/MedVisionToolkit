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

from typing import Union, List
import torch
import torch.nn as nn

from medtk.model.nnModules import ComponentModule, BlockModule
from medtk.model.nd import MaxPoolNd
from medtk.model.blocks import ConvNormAct


class Input(BlockModule):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, padding=1, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvNormAct(dim, in_channels, out_channels, kernel_size, padding=padding))
            else:
                layers.append(ConvNormAct(dim, out_channels, out_channels, kernel_size, padding=padding))
        self.in_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.in_conv(x)


class Down(BlockModule):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, padding=1, num_layers=2, conv_pool=False):
        super().__init__()
        stride = 2 if conv_pool else 1
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvNormAct(dim, in_channels, out_channels, kernel_size, padding=padding, stride=stride))
            else:
                layers.append(ConvNormAct(dim, out_channels, out_channels, kernel_size, padding=padding))
        if conv_pool:
            self.down_conv = nn.Sequential(*layers)
        else:
            self.down_conv = nn.Sequential(MaxPoolNd(dim)(2), *layers)

    def forward(self, x):
        return self.down_conv(x)


class UNetEncoder(ComponentModule):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 base_width=16,
                 stages=5,
                 out_indices=(0, 1, 2, 3, 4),
                 bilinear=True):
        super(UNetEncoder, self).__init__()
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        assert max(out_indices) < stages, "max out_index must smaller than stages"
        self.dim = dim
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.base_width = base_width
        self.stages = stages
        self.bilinear = bilinear

        self.in_block, self.downs = self.init_layers()
        self.init_weights()

    def init_layers(self):
        in_block = Input(self.dim, self.in_channels, self.base_width)  # the first stage

        downs = []
        for i in range(self.stages - 1):
            # base_width = 64, i = 0-3
            # in planes  : 64 -> 128 -> 256 -> 512
            # out planes : 128 -> 256 -> 512 -> 1024
            planes = self.base_width * pow(2, i)
            downs.append(Down(self.dim, planes, planes * 2, conv_pool=not self.bilinear))

        downs = nn.ModuleList(downs)
        return in_block, downs

    def init_weights(self):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        # e.g.
        # x, torch.Size([1, 1, 32, 32])
        x = self.in_block(x)  # torch.Size([1, 64, 32, 32])

        outs_down = [x]  # 0, torch.Size([1, 64, 32, 32])
        for i in range(self.stages - 1):
            x = self.downs[i](x)
            outs_down.append(x)
        # 1, torch.Size([1, 128, 16, 16])
        # 2, torch.Size([1, 256, 8, 8])
        # 3, torch.Size([1, 512, 4, 4])
        # 4, torch.Size([1, 1024, 2, 2])

        outs = []
        for i in self.out_indices:
            outs.append(outs_down[i])
        return outs


class LightUNetEncoder(UNetEncoder):
    def init_layers(self):
        inc = Input(self.dim, self.in_channels, self.base_width)

        downs = []
        for i in range(self.stages - 1):
            # base_width = 64, i = 0-3
            # in planes  : 64 -> 128 -> 256 -> 512
            # out planes : 128 -> 256 -> 512 -> 512 (same with previous layer)
            planes = self.base_width * pow(2, i)
            if i == self.stages - 2:
                downs.append(Down(self.dim, planes, planes))
            else:
                downs.append(Down(self.dim, planes, planes * 2))

        downs = nn.ModuleList(downs)
        return inc, downs


class FlexUNetEncoder(ComponentModule):
    def __init__(self,
                 dim: int,
                 in_channels: int,
                 base_width=16,
                 stages=5,
                 out_indices=(0, 1, 2, 3, 4)):
        super(FlexUNetEncoder, self).__init__()
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        assert max(out_indices) < stages, "max out_index must smaller than stages"
        self.dim = dim
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.base_width = base_width
        self.stages = stages

        self.inc, self.downs = self.init_layers()
        self.init_weights()


if __name__ == "__main__":
    model = UNetEncoder(2, 3, base_width=64, stages=4, out_indices=(0, 1, 2, 3))
    model.print_model_params()
    print(model)
    data = torch.rand((1, 3, 32, 32))
    outs = model(data)
    for o in outs:
        print(o.shape)
