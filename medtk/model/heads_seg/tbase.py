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
import torch.nn.functional as F


class BaseSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, loss_cls, metrics):
        super(BaseSegHead, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        if isinstance(metrics, dict):
            metrics = [metrics]
        self.in_channels = in_channels
        self.out_channels = num_classes + 1
        self.criterion = loss_cls
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self.base_criterion = nn.CrossEntropyLoss()
        self.layers = None
        self.init_layers()

    def init_layers(self):
        layers = []
        for in_channel in self.in_channels:
            layers.append(self._make_single_out(in_channel, self.out_channels))
        self.layers = nn.ModuleList(layers)

    def _make_single_out(self, in_channels, out_channels):
        # return nn.Sequential( not work
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        assert len(self.in_channels) == len(x), 'backbone/neck outs not match head ins'
        outs = []
        for i in range(len(self.in_channels)):
            outs.append(self.layers[i](x[i]))
        return outs

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output, ground_truth)
            metrics.update(one_metric)
        return metrics

    def loss(self, net_output, ground_truth):
        return self.criterion(net_output, ground_truth)

    def loss_reference(self, net_output, ground_truth):
        return self.base_criterion(net_output, ground_truth)


@HEADS.register_module
class BaseSegHead3dTest(BaseSegHead):
    def _make_single_out(self, in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, kernel_size=1)


@HEADS.register_module
class StackedSegHead(BaseSegHead):
    def __init__(self, stages=1, *args, **kwargs):
        self.stages = stages
        super().__init__(*args, **kwargs)

    def _make_single_out(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=4, dilation=2),
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._make_layers()

    def _make_layers(self):
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + out)
        return out


@HEADS.register_module
class RefineSegHead(BaseSegHead):
    def __init__(self, stages=1, *args, **kwargs):
        self.stages = stages
        super().__init__(*args, **kwargs)

    def _make_single_out(self, in_channels, out_channels):
        return ResBlock(in_channels, out_channels)


class ResEdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._make_layers()

    def _make_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
        )] * self.out_channels)
        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(1),
        )] * self.out_channels)
        self.conv3 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )] * self.out_channels)
        self.conv4 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
        )] * self.out_channels)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5, padding=4, dilation=2, bias=False)

    def forward(self, x, edge):
        # print(x.shape, edge.shape, self.in_channels)
        out = []
        x_slice = torch.split(x, [1] * x.shape[1], dim=1)
        for i in range(x.shape[1]):
            x_i = x_slice[i]
            x_i = self.relu(self.conv1[i](torch.cat([x_i, edge], dim=1)) + x_i)
            x_i = self.relu(self.conv2[i](torch.cat([x_i, edge], dim=1)) + x_i)
            x_i = self.relu(self.conv3[i](torch.cat([x_i, edge], dim=1)) + x_i)
            x_i = self.relu(self.conv4[i](torch.cat([x_i, edge], dim=1)) + x_i)
            out.append(self.conv5(x_i))
        x = torch.cat(out, dim=1)
        return x


@HEADS.register_module
class ResEdgeSegHead(BaseSegHead):
    def __init__(self, edge_channels, **kwargs):
        self.edge_channels = edge_channels
        self.layers = None
        self.edge_convs = None
        super().__init__(**kwargs)

    def init_layers(self):
        layers = []
        edge_convs = []
        for in_channel in self.in_channels:
            layers.append(self._make_single_out(in_channel, self.out_channels))
            edge_convs.append(ResEdgeBlock(1 + self.edge_channels, self.out_channels))
        self.layers = nn.ModuleList(layers)
        self.edge_convs = nn.ModuleList(edge_convs)

    def _make_single_out(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, edge=None, *args, **kwargs):
        assert len(self.in_channels) == len(x), 'backbone/neck outs not match head ins'
        outs = []
        for i in range(len(self.in_channels)):
            e = F.avg_pool2d(edge, 2 ** i)
            out = self.layers[i](x[i])
            outs.append(self.edge_convs[i](out, e))
        return outs

    # def metric(self, net_output, ground_truth):
    #     metrics = {}
    #     for metric in self.metrics:
    #         one_metric = metric(net_output, ground_truth)
    #         metrics.update(one_metric)
    #     return metrics
    #
    # def loss(self, prediction, ground_truth):
    #     return self.criterion(prediction, ground_truth)
    #
    # def loss_reference(self, prediction, ground_truth):
    #     return self.base_criterion(prediction, ground_truth)


if __name__ == "__main__":
    import torch

    x = torch.rand([1, 16, 32, 32])
    h = StackedSegHead(1, [16], 2, dict(type='CrossEntropyLoss'), [dict(type='Dice')])
    print(h)
    outs = h([x])
    for out in outs:
        print(out.shape)