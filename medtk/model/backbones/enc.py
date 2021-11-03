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
from medtk.model.blocks import ConvNormAct, VResConvNormAct, \
    BasicBlockNd, BottleneckNd, \
    SEBasicBlockNd, SEBottleneckNd


class ConvLayer(BlockModule):
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                self.blocks.append(ConvNormAct(dim, in_channels, out_channels, kernel_size=3, padding=1, stride=stride))
            else:
                self.blocks.append(ConvNormAct(dim, out_channels, out_channels, kernel_size=3, padding=1))
            self.blocks.append(ConvNormAct(dim, out_channels, out_channels, kernel_size=3, padding=1))

        for i, m in enumerate(self.blocks):
            self.add_module(str(i), m)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class VResConvLayer(BlockModule):
    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                self.blocks.append(ConvNormAct(dim, in_channels, out_channels, kernel_size=3, padding=1, stride=stride))
            else:
                self.blocks.append(VResConvNormAct(dim, out_channels, out_channels, kernel_size=3, padding=1))

        for i, m in enumerate(self.blocks):
            self.add_module(str(i), m)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class ResidualLayer(BlockModule):
    BLOCK = BasicBlockNd

    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False):
        super(ResidualLayer, self).__init__()
        self.num_blocks = num_blocks
        blocks = nn.ModuleList([self.BLOCK(dim,
                                           in_planes=in_channels,
                                           planes=out_channels,
                                           stride=stride,
                                           bias=bias,
                                           dilation=dilation,
                                           groups=groups,
                                           width_per_group=width_per_group)
                                ])
        in_planes = out_channels * self.BLOCK.expansion
        for i in range(1, num_blocks):
            blocks.append(
                self.BLOCK(dim,
                           in_planes=in_planes,
                           planes=out_channels,
                           stride=1,
                           bias=bias,
                           dilation=dilation,
                           groups=groups,
                           width_per_group=width_per_group))

        for i, m in enumerate(blocks):
            self.add_module(str(i), m)

    def forward(self, x):
        for i in range(self.num_blocks):
            layer = getattr(self, str(i))
            x = layer(x)
        return x


class ResidualBottleneckLayer(ResidualLayer):
    BLOCK = BottleneckNd

    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False):
        out_channels = out_channels // self.BLOCK.expansion
        super(ResidualBottleneckLayer, self).__init__(dim,
                                                      in_channels,
                                                      out_channels,
                                                      stride,
                                                      num_blocks,
                                                      groups,
                                                      width_per_group,
                                                      dilation,
                                                      bias)


class SEResidualLayer(ResidualLayer):
    BLOCK = SEBasicBlockNd

    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False):
        super(SEResidualLayer, self).__init__(dim,
                                              in_channels,
                                              out_channels,
                                              stride,
                                              num_blocks,
                                              groups,
                                              width_per_group,
                                              dilation,
                                              bias)


class SEResidualBottleneckLayer(ResidualLayer):
    BLOCK = SEBottleneckNd

    def __init__(self,
                 dim,
                 in_channels,
                 out_channels,
                 stride,
                 num_blocks,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False):
        out_channels = out_channels // self.BLOCK.expansion
        super(SEResidualBottleneckLayer, self).__init__(dim,
                                                        in_channels,
                                                        out_channels,
                                                        stride,
                                                        num_blocks,
                                                        groups,
                                                        width_per_group,
                                                        dilation,
                                                        bias)


class Encoder(ComponentModule):
    """
    support list:
        - Vanilla UNet
        - ResNet
        - ResNeXt
        -
    """
    LAYERS = {
        'conv':     (ConvLayer, 1),  # UNet, VNet
        'v_conv':   (VResConvLayer, 1),  # VBNet
        'res':      (ResidualLayer, 1),  # ResNet 18, 34
        'b_res':    (ResidualBottleneckLayer, 4),  # ResNet or ResNeXt ge than 50
        'se_res':   (SEResidualLayer, 1),  # SEResNet 18, 34
        'se_b_res': (SEResidualBottleneckLayer, 4),
    }

    def __init__(self,
                 dim: int,
                 in_channels: int,
                 features=(16, 32, 64, 128),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 num_blocks=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 layer_cfg: dict = {'type': 'conv'},
                 groups=1,
                 width_per_group=64,
                 first_conv=(64, 7, 1),
                 downsample=False):
        super(Encoder, self).__init__()
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        assert max(out_indices) < len(strides), "max out_index must smaller than stages"
        assert len(strides) == len(num_blocks) == len(features)

        self.first_features, self.first_kernel, self.first_stride = first_conv
        self.downsample = downsample

        self.dim = dim
        self.in_channels = in_channels
        self.features = features
        self.strides = strides
        self.dilations = dilations
        self.num_blocks = num_blocks
        self.out_indices = out_indices
        self.stages = len(self.strides)
        self.groups = groups
        self.width_per_group = width_per_group

        self.layer_type = layer_cfg.pop('type')
        self.layer_cfg = layer_cfg
        assert self.layer_type in self.LAYERS.keys()
        self.layer, self.expansion = self.LAYERS[self.layer_type]

        self.conv1 = self.build_conv(dim, self.in_channels,
                                     self.first_features,
                                     kernel_size=self.first_kernel,
                                     stride=self.first_stride,
                                     padding=self.first_kernel // 2,
                                     bias='res' not in self.layer_type)
        self.bn1 = self.build_norm(self.dim, self.first_features)
        self.relu = self.build_act()
        self.maxpool = MaxPoolNd(self.dim)(kernel_size=3, stride=2, padding=1)

        self.init_layers()
        # self.init_weights()

    def init_layers(self):
        layers = nn.ModuleList()
        in_planes = self.first_features
        for i in range(self.stages):
            layer_name = 'layer{}'.format(i + 1)
            layer = self.layer(
                self.dim,
                in_planes,
                self.features[i],
                stride=self.strides[i],
                num_blocks=self.num_blocks[i],
                groups=self.groups,
                width_per_group=self.width_per_group,
                dilation=self.dilations[i])

            in_planes = self.features[i]
            self.add_module(layer_name, layer)
            layers.append(layer)

        return layers

    def init_weights(self):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        if self.downsample:
            x = self.maxpool(x)

        outs = []
        for i in range(self.stages):
            layer_name = 'layer{}'.format(i + 1)
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


if __name__ == "__main__":
    from medtk.runner.checkpoint import load_checkpoint

    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    init_seed(666)

    # ResNet18 = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(64, 128, 256, 512),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(2, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     first_conv=(64, 7, 2),
    #     layer_type='res',
    #     downsample=True
    # )
    # load_checkpoint(ResNet18, 'https://download.pytorch.org/models/resnet18-5c106cde.pth')
    #
    # ResNet34 = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(64, 128, 256, 512),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(3, 4, 6, 3),
    #     out_indices=(0, 1, 2, 3),
    #     first_conv=(64, 7, 2),
    #     layer_type='res',
    #     downsample=True
    # )
    # load_checkpoint(ResNet34, 'https://download.pytorch.org/models/resnet34-333f7ec4.pth')
    #
    # ResNet50 = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(256, 512, 1024, 2048),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(3, 4, 6, 3),
    #     out_indices=(0, 1, 2, 3),
    #     first_conv=(64, 7, 2),
    #     layer_type='b_res',
    #     downsample=True
    # )
    # load_checkpoint(ResNet50, 'https://download.pytorch.org/models/resnet50-19c8e357.pth')
    #
    # ResNeXt50_32x4 = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(256, 512, 1024, 2048),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(3, 4, 6, 3),
    #     out_indices=(0, 1, 2, 3),
    #     groups=32,
    #     width_per_group=4,
    #     first_conv=(64, 7, 2),
    #     layer_type='b_res',
    #     downsample=True
    # )
    # load_checkpoint(ResNeXt50_32x4, 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth')

    # SEResNet50 = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(256, 512, 1024, 2048),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(3, 4, 6, 3),
    #     out_indices=(0, 1, 2, 3),
    #     first_conv=(64, 7, 2),
    #     layer_type='se_b_res',
    #     downsample=True
    # )
    # load_checkpoint(SEResNet50, 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth')

    # UNet = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(32, 64, 128, 256),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(1, 1, 1, 1),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='conv'
    # )
    # model = UNet

    # VNet = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(32, 64, 128, 256),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(1, 2, 3, 4),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='v_conv'
    # )
    # model = VNet

    # TVNet = Encoder(
    #     dim=2,
    #     in_channels=3,
    #     features=(32, 64, 128, 256),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(2, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='v_conv'
    # )
    # model = TVNet

    DeeplungResNet18 = Encoder(
        dim=3,
        in_channels=1,
        features=(32, 64, 64, 64),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        num_blocks=(2, 2, 3, 3),
        out_indices=(0, 1, 2, 3),
        first_conv=(24, 7, 2),
        layer_cfg=dict(
            type='res',
        )
        # groups=32,
        # width_per_group=4,
    )
    model = DeeplungResNet18

    # ResNet18 = Encoder(
    #     dim=3,
    #     in_channels=1,
    #     features=(16, 32, 64, 128),
    #     strides=(1, 2, 2, 2),
    #     dilations=(1, 1, 1, 1),
    #     num_blocks=(2, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     first_conv=(64, 7, 2),
    #     layer_type='res',
    #     downsample=True
    # )
    # model = ResNet18

    print(model)

    model.print_model_params()
    data = torch.ones((1, 1, 96, 96, 96))
    outs = model(data)
    for o in outs:
        print(o.shape)
        # print(torch.sum(o))