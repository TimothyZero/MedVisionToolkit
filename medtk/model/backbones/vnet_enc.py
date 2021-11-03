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

from medtk.model.nnModules import BlockModule, ComponentModule
from medtk.model.blocks import ConvNormAct


class BottleConvBnRelu(BlockModule):
    def __init__(self, dim, channels, ratio, do_act=True, bias=True):
        super(BottleConvBnRelu, self).__init__()
        self.conv1 = ConvNormAct(dim, channels, channels // ratio, kernel_size=1, padding=0, do_act=True, bias=bias)
        self.conv2 = ConvNormAct(dim, channels // ratio, channels // ratio, kernel_size=3, padding=1, do_act=True, bias=bias)
        self.conv3 = ConvNormAct(dim, channels // ratio, channels, kernel_size=1, padding=0, do_act=do_act, bias=bias)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        return out


class ResidualBlock(BlockModule):
    def __init__(self, dim, channels, kernel_size, padding, num_layers):
        super(ResidualBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                layers.append(ConvNormAct(dim, channels, channels, kernel_size, padding, do_act=True))
            else:
                layers.append(ConvNormAct(dim, channels, channels, kernel_size, padding, do_act=False))
        self.ops = nn.Sequential(*layers)
        self.act = self.build_act()

    def forward(self, x):
        output = self.ops(x)
        return self.act(x + output)


class BottleResidualBlock(BlockModule):
    def __init__(self, dim, channels, ratio, num_layers):
        super(BottleResidualBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            if i != num_layers - 1:
                layers.append(BottleConvBnRelu(dim, channels, ratio, True))
            else:
                layers.append(BottleConvBnRelu(dim, channels, ratio, False))
        self.ops = nn.Sequential(*layers)
        self.act = self.build_act()

    def forward(self, x):
        output = self.ops(x)
        return self.act(x + output)


class Input(BlockModule):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, padding=1, num_layers=1):
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


class Down(nn.Module):
    def __init__(self, dim, in_channels, num_layers, use_bottle_neck=False):
        super(Down, self).__init__()
        out_channels = in_channels * 2
        if use_bottle_neck:
            block = BottleResidualBlock(dim, out_channels, ratio=4, num_layers=num_layers)
        else:
            block = ResidualBlock(dim, out_channels, kernel_size=3, padding=1, num_layers=num_layers)
        self.block = nn.Sequential(
            ConvNormAct(dim, in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            block
        )

    def forward(self, x):
        out = self.block(x)
        return out


class VBNetEncoder(ComponentModule):
    def __init__(self,
                 dim,
                 in_channels,
                 base_width=16,
                 stages=5,
                 out_indices=(0, 1, 2, 3, 4),
                 use_bottle_neck=False):
        super(VBNetEncoder, self).__init__()
        self.arch_settings = {
            5: (2, 2, 3, 3, 3)
        }
        self.dim = dim
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.base_width = base_width
        self.stages = stages
        self.use_bottle_neck = use_bottle_neck

        self.in_block, self.downs = self.init_layers()
        self.init_weights()

    def init_layers(self):
        in_block = Input(self.dim, self.in_channels, self.base_width)

        downs = []
        for i in range(self.stages - 1):
            # base_width = 64, i = 0-3, 64 -> 128 -> 256 -> 512
            planes = self.base_width * pow(2, i)
            downs.append(Down(self.dim, in_channels=planes, num_layers=min(3, i + 1), use_bottle_neck=self.use_bottle_neck))
            # print('d', planes, planes * 2)

        downs = nn.ModuleList(downs)
        return in_block, downs

    def init_weights(self, bn_std=0.02):
        # pass
        # for m in self.modules():
        #     if isinstance(m, ConvNd(self.dim)):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, BatchNormNd(self.dim)):
        #         nn.init.normal_(m.weight, 1.0, bn_std)
        #         m.bias.data.zero_()

        #     classname = m.__class__.__name__
        #     if 'Conv' in classname:
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif 'BatchNorm' in classname:
        #         m.weight.data.normal_(1.0, bn_std)
        #         m.bias.data.zero_()
        #     elif 'Linear' in classname:
        #         nn.init.kaiming_normal(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.in_block(x)

        outs_down = [x]
        for i in range(self.stages - 1):
            x = self.downs[i](x)
            outs_down.append(x)

        outs = []
        for i in self.out_indices:
            outs.append(outs_down[i])
            # print('in', outs_down[i].shape)

        return outs


if __name__ == "__main__":
    model = VBNetEncoder(
        dim=3,
        in_channels=3,
        base_width=64,
        stages=4,
        out_indices=(0, 1, 2, 3)
    )
    print(model)
    model.print_model_params()
    data = torch.rand((1, 3, 32, 32, 32))
    outs = model(data)
    for o in outs:
        print(o.shape)