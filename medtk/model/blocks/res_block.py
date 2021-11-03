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
from torch import nn
from ..nnModules import BlockModule


class BasicBlockNd(BlockModule):
    expansion = 1

    def __init__(self,
                 dim,
                 in_planes,
                 planes,
                 stride=1,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(BasicBlockNd, self).__init__(conv_cfg=None, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if groups != 1 or width_per_group != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dim = dim
        self.conv1 = self.build_conv(self.dim, in_planes, planes, stride=stride, kernel_size=3, padding=1,
                                     bias=bias)
        self.bn1 = self.build_norm(self.dim, planes)
        self.relu = self.build_act()
        self.conv2 = self.build_conv(self.dim, planes, planes, stride=1, kernel_size=3, padding=1,
                                     bias=bias)
        self.bn2 = self.build_norm(self.dim, planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                self.build_conv(dim, in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=bias),
                self.build_norm(dim, planes * self.expansion),
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckNd(BlockModule):
    expansion = 4

    def __init__(self,
                 dim,
                 in_planes,
                 planes,
                 stride=1,
                 groups=1,
                 width_per_group=64,
                 dilation=1,
                 bias=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(BottleneckNd, self).__init__(conv_cfg=None, norm_cfg=norm_cfg, act_cfg=act_cfg)
        width = int(planes * (width_per_group / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.dim = dim
        self.conv1 = self.build_conv(self.dim, in_planes, width, kernel_size=1, stride=1,
                                     bias=bias)
        self.bn1 = self.build_norm(self.dim, width)
        self.conv2 = self.build_conv(self.dim, width, width, stride=stride, kernel_size=3, padding=dilation,
                                     groups=groups, dilation=dilation,
                                     bias=False, conv_cfg=conv_cfg)
        self.bn2 = self.build_norm(self.dim, width)
        self.conv3 = self.build_conv(self.dim, width, planes * self.expansion, kernel_size=1, stride=1,
                                     bias=bias)
        self.bn3 = self.build_norm(self.dim, planes * self.expansion)
        self.relu = self.build_act()
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                self.build_conv(dim, in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=bias),
                self.build_norm(dim, planes * self.expansion),
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
