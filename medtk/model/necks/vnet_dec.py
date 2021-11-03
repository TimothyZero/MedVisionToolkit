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

from medtk.model.nnModules import BlockModule, ComponentModule
from medtk.model.nd import ConvNd, BatchNormNd, ConvTransposeNd
from medtk.model.blocks import ConvNormAct


class BottleConvBnRelu(BlockModule):
    def __init__(self, dim, channels, ratio, do_act=True, bias=True):
        super(BottleConvBnRelu, self).__init__()
        self.conv1 = ConvNormAct(dim, channels, channels // ratio, kernel_size=1, padding=0, do_act=True, bias=bias)
        self.conv2 = ConvNormAct(dim, channels // ratio, channels // ratio, kernel_size=3, padding=1, do_act=True,
                                 bias=bias)
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


class UpBlock(BlockModule):
    def __init__(self, dim, in_channels, out_channels, num_layers, use_bottle_neck=False):
        super(UpBlock, self).__init__()
        self.dim = dim
        self.up_conv = ConvTransposeNd(dim)(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
        self.up_bn = self.build_norm(dim, out_channels // 2)
        self.up_act = self.build_act()
        if use_bottle_neck:
            self.rblock = BottleResidualBlock(dim, out_channels, ratio=4, num_layers=num_layers)
        else:
            self.rblock = ResidualBlock(dim, out_channels, kernel_size=3, padding=1, num_layers=num_layers)

    def forward(self, x, skip):
        x = self.up_act(self.up_bn(self.up_conv(x)))

        if self.dim == 3:
            # input is CDHW
            diffZ = skip.size()[2] - x.size()[2]
            diffY = skip.size()[3] - x.size()[3]
            diffX = skip.size()[4] - x.size()[4]

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])
            # print(diffX, diffY, diffZ)
        elif self.dim == 2:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        else:
            raise NotImplementedError

        x = torch.cat((x, skip), 1)
        x = self.rblock(x)
        return x


class VBNetDecoder(ComponentModule):
    """volumetric secmentation network """

    def __init__(self, dim, in_channels, out_indices=(0,), use_bottle_neck=False):
        super(VBNetDecoder, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.dim = dim

        ups = []
        for i, planes in enumerate(reversed(in_channels[:-1])):
            # in_channels = [64, 128, 256, 512, 1024] // [16, 32, 64, 128, 256] // [1, 2, 4, 8, 16]
            if i == 0:
                ups.append(UpBlock(dim, planes * 2, planes * 2, min(3, 4 - i), use_bottle_neck))
            else:
                ups.append(UpBlock(dim, planes * 4, planes * 2, min(3, 4 - i), use_bottle_neck))
            # print('u', planes)

        self.ups = nn.ModuleList(ups)
        self.init_weights()

    def init_weights(self, bn_std=0.02):
        pass
        # seem worse than default initial
        # for m in self.modules():
        #     if isinstance(m, ConvNd(self.dim)):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, BatchNormNd(self.dim)):
        #         nn.init.normal_(m.weight, 1.0, bn_std)
        #         m.bias.data.zero_()

        # # self.apply()
        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if 'Conv' in classname:
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero()
        #     elif 'BatchNorm' in classname:
        #         m.weight.data.normal(1.0, bn_std)
        #         m.bias.data.zero()
        #     elif 'Linear' in classname:
        #         nn.init.kaiming_normal(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        assert len(self.in_channels) == len(x)
        outs_up = [x[-1]]
        for i in range(len(self.in_channels) - 1):
            in1, in2 = outs_up[-1], x[-i - 2]
            # print(i, in1.shape, in2.shape)
            y = self.ups[i](in1, in2)
            # print('dec', y.shape)
            outs_up.append(y)

        outs_up.reverse()

        outs = []
        for i in self.out_indices:
            outs.append(outs_up[i])
            # print(outs_up[i].shape)
        return outs


if __name__ == "__main__":
    import torch

    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    init_seed(666)

    model = VBNetDecoder(3, [16, 32, 64, 128], (0, 1, 2, 3))
    print(model)
    model.print_model_params()

    inputs = [
        torch.ones((1, 16, 32, 32, 32)),
        torch.ones((1, 32, 16, 16, 16)),
        torch.ones((1, 64, 8, 8, 8)),
        torch.ones((1, 128, 4, 4, 4)),
    ]
    outs = model(inputs)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))
