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
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from medtk.model.nnModules import ComponentModule, BlockModule
from medtk.model.nd import ConvTransposeNd, ConvNd, DropoutNd
from medtk.model.blocks import ConvNormAct, ConvTransposeNormAct, BasicBlockNd


class ConcatLayer(BlockModule):
    S_MODE = False

    def __init__(self,
                 dim,
                 deeper_channels,
                 lower_channels,
                 out_channels,
                 stride,
                 num_blocks=2,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 bias=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dim = dim
        self.mid_channels = lower_channels if self.S_MODE else deeper_channels
        self.deeper_conv = ConvTransposeNormAct(dim, deeper_channels, self.mid_channels, kernel_size=stride,
                                                stride=stride)

        fusion_conv = [ConvNormAct(dim, self.mid_channels + lower_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1, bias=bias)]
        for i in range(1, num_blocks):
            fusion_conv.append(
                ConvNormAct(dim, out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        self.fusion_conv = nn.Sequential(*fusion_conv)

    def forward(self, x_deeper, x_lower):
        mid = self.deeper_conv(x_deeper)
        if self.dim == 3:
            # input is CDHW
            diffZ = x_lower.size()[2] - mid.size()[2]
            diffY = x_lower.size()[3] - mid.size()[3]
            diffX = x_lower.size()[4] - mid.size()[4]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
        elif self.dim == 2:
            diffY = x_lower.size()[2] - mid.size()[2]
            diffX = x_lower.size()[3] - mid.size()[3]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        else:
            raise NotImplementedError
        output = self.fusion_conv(torch.cat([mid, x_lower], dim=1))
        return output


class ResConcatLayer(BlockModule):
    def __init__(self,
                 dim,
                 deeper_channels,
                 lower_channels,
                 out_channels,
                 stride,
                 num_blocks=3,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dim = dim
        self.mid_channels = deeper_channels
        self.deeper_conv = ConvTransposeNormAct(dim, deeper_channels, self.mid_channels, kernel_size=stride,
                                                stride=stride)

        fusion_conv = [BasicBlockNd(dim, self.mid_channels + lower_channels, out_channels, stride=1, bias=bias)]
        for i in range(1, num_blocks):
            fusion_conv.append(
                BasicBlockNd(dim, out_channels, out_channels, stride=1, bias=bias)
            )
        self.fusion_conv = nn.Sequential(*fusion_conv)

    def forward(self, x_deeper, x_lower):
        mid = self.deeper_conv(x_deeper)
        if self.dim == 3:
            # input is CDHW
            diffZ = x_lower.size()[2] - mid.size()[2]
            diffY = x_lower.size()[3] - mid.size()[3]
            diffX = x_lower.size()[4] - mid.size()[4]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
        elif self.dim == 2:
            diffY = x_lower.size()[2] - mid.size()[2]
            diffX = x_lower.size()[3] - mid.size()[3]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        else:
            raise NotImplementedError
        output = self.fusion_conv(torch.cat([mid, x_lower], dim=1))
        return output


class SConcatLayer(ConcatLayer):
    S_MODE = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AddLayer(BlockModule):
    def __init__(self,
                 dim,
                 deeper_channels,
                 lower_channels,
                 out_channels,
                 stride,
                 num_blocks=1,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dim = dim
        if deeper_channels != out_channels:
            self.deeper_conv = nn.Sequential(
                ConvNd(dim)(deeper_channels, out_channels, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=stride, mode='nearest'),
            )
        else:
            self.deeper_conv = nn.Upsample(scale_factor=stride, mode='nearest')
        self.lower_conv = ConvNd(dim)(lower_channels, out_channels, kernel_size=1, stride=1)

        fusion_conv = [ConvNd(dim)(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        for i in range(1, num_blocks):
            fusion_conv.append(
                ConvNormAct(dim, out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        self.fusion_conv = nn.Sequential(*fusion_conv)

    def forward(self, x_deeper, x_lower):
        x_deeper = self.deeper_conv(x_deeper)
        x_lower = self.lower_conv(x_lower)
        if self.dim == 3:
            # input is CDHW
            diffZ = x_lower.size()[2] - x_deeper.size()[2]
            diffY = x_lower.size()[3] - x_deeper.size()[3]
            diffX = x_lower.size()[4] - x_deeper.size()[4]

            x_deeper = F.pad(x_deeper, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2,
                                        diffZ // 2, diffZ - diffZ // 2])
        elif self.dim == 2:
            diffY = x_lower.size()[2] - x_deeper.size()[2]
            diffX = x_lower.size()[3] - x_deeper.size()[3]

            x_deeper = F.pad(x_deeper, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        else:
            raise NotImplementedError

        output = self.fusion_conv(torch.add(x_deeper, x_lower))
        return output


class Decoder(ComponentModule):
    LAYERS = {
        'concat': ConcatLayer,
        'res_concat': ResConcatLayer,
        'sconcat': SConcatLayer,
        'add': AddLayer,
    }

    def __init__(self,
                 dim: int,
                 in_channels: Union[list, tuple] = (16, 32, 64, 128),
                 out_channels: Union[list, tuple] = (16, 32, 64, 128),
                 out_indices: Union[list, tuple] = (0, 1, 2, 3),
                 strides=(1, 2, 2, 2),
                 num_blocks=(1, 1, 1, 1),
                 layer_cfg: dict = {'type': 'concat'},
                 dropout=False,
                 with_coord=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(Decoder, self).__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_indices = out_indices
        self.strides = strides
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.with_coord = with_coord

        self.layer_type = layer_cfg.pop('type')
        self.layer_cfg = layer_cfg
        self.layer = self.LAYERS[self.layer_type]
        self.conv_last = self.layer_type == 'add' or self.in_channels[-1] != self.out_channels[-1]

        if self.conv_last:
            self.conv1 = self.build_conv(self.dim,
                                         self.in_channels[-1],
                                         self.out_channels[-1],
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
            self.conv2 = self.build_conv(self.dim,
                                         self.out_channels[-1],
                                         self.out_channels[-1],
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        if self.dropout:
            self.drop = DropoutNd(self.dim)(p=0.5, inplace=False)

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        layers = nn.ModuleList()
        for i in range(len(self.in_channels) - 1, 0, -1):
            if i == 1 and self.with_coord:
                addition = 3
            else:
                addition = 0
            layer_name = f'layer{i + 1}+{i}'
            layer = self.layer(self.dim,
                               deeper_channels=self.out_channels[i],
                               lower_channels=self.in_channels[i - 1] + addition,
                               out_channels=self.out_channels[i - 1],
                               stride=self.strides[i],
                               num_blocks=self.num_blocks[i],
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg,
                               **self.layer_cfg.copy())
            self.add_module(layer_name, layer)
            layers.append(layer)

        return layers

    def _init_weights(self, bn_std=0.02):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, bn_std)
                m.bias.data.zero_()

    def forward(self, inputs):
        assert len(self.in_channels) == len(inputs)

        outs_up = [inputs[-1]]

        if self.conv_last:
            outs_up[0] = self.conv1(outs_up[0])

        for i in range(len(self.in_channels) - 1):
            x_deeper, x_lower = outs_up[-1], inputs[-i - 2]
            if i == (len(self.in_channels) - 2) and self.with_coord:
                coord = torch.stack(torch.meshgrid(*map(torch.arange, x_lower.shape[2:])))
                coord = coord.repeat(x_lower.shape[0], 1, 1, 1, 1).to(x_lower.device)
                x_lower = torch.cat((x_lower, coord), 1)
            layer_name = f'layer{len(self.in_channels) - i}+{len(self.in_channels) - i - 1}'
            layer = getattr(self, layer_name)
            y = layer(x_deeper, x_lower)
            outs_up.append(y)

        if self.conv_last:
            outs_up[0] = self.conv2(outs_up[0])

        outs_up.reverse()

        outs = []
        for i in self.out_indices:
            if self.dropout:
                outs.append(self.drop(outs_up[i]))
            else:
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

    # FPN = Decoder(
    #     dim=2,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(64, 64, 64, 64),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='add')
    # model = FPN

    # UNet = Decoder(
    #     dim=2,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(16, 32, 64, 128),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='concat')
    # model = UNet

    # VNet = Decoder(
    #     dim=3,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(32, 64, 128, 128),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='concat')
    # model = VNet

    Deeplung_dec = Decoder(
        dim=3,
        in_channels=(64, 64, 64),
        out_channels=(128, 64, 64),
        strides=(1, 2, 2),
        num_blocks=(3, 3, 3),
        out_indices=(0,),
        layer_cfg=dict(
            type='res_concat',
            bias=True,
        ),
        dropout=True,
    )
    model = Deeplung_dec

    print(model)
    model.print_model_params()

    inputs = [
        # torch.ones((1, 32, 48, 48, 48)),
        torch.ones((1, 64, 24, 24, 24)),
        torch.ones((1, 64, 12, 12, 12)),
        torch.ones((1, 64, 6, 6, 6)),
    ]

    outs = model(inputs)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))
