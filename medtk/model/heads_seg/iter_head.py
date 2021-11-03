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

from medtk.model.backbones.unet_enc import Input, Down
from medtk.model.necks.unet_dec import Up
from medtk.model.nnModules import ComponentModule
from medtk.model.nd import ConvNd


class MiniUNet(ComponentModule):
    def __init__(self, dim, in_channels, out_channels, n_classes, bilinear):
        super(MiniUNet, self).__init__()
        self.dim = dim
        self.in_channels = in_channels

        self.inc = Input(dim, in_channels, out_channels)
        self.down1 = Down(dim, out_channels, out_channels * 2)
        self.down2 = Down(dim, out_channels * 2, out_channels * 4)
        self.down3 = Down(dim, out_channels * 4, out_channels * 8)
        self.up1 = Up(dim, out_channels * 8, out_channels * 4, bilinear)
        self.up2 = Up(dim, out_channels * 4, out_channels * 2, bilinear)
        self.up3 = Up(dim, out_channels * 2, out_channels, bilinear)
        self.outc = ConvNd(self.dim)(out_channels, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return x1, x, logits
        # return x1, x


class IterHead(ComponentModule):
    def __init__(self, dim, in_channels, num_classes, loss_cls, metrics):
        super(IterHead, self).__init__()
        self.dim = dim
        self.in_channels = in_channels[0]
        self.out_channels = num_classes
        assert self.out_channels > 1
        self.criterion = loss_cls
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self.base_criterion = nn.CrossEntropyLoss()

        self.iterations = 2
        self.model_miniunet = nn.ModuleList(
            [MiniUNet(dim, self.in_channels * 2, self.in_channels, num_classes, bilinear=True)] * self.iterations)

    def forward(self, x_bk, x_nk):
        x1, x2 = x_bk[0], x_nk[0]
        for i in range(self.iterations):
            x = torch.cat([x1, x2], dim=1)
            # print(x.shape)
            _, x2, logits = self.model_miniunet[i](x)
            # print(_.shape, x2.shape)
        return [logits]

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output, ground_truth)
            metrics.update(one_metric)
        return metrics

    def loss(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)

    def loss_reference(self, prediction, ground_truth):
        return self.base_criterion(prediction, ground_truth)


if __name__ == '__main__':
    # pass
    # m = MiniUNet(3, 2, 16, 32, 2)
    # data = torch.rand(1, 16, 32, 32, 32)
    # ps = m(data)
    # [print(p.shape) for p in ps]

    data = [torch.rand(1, 16, 32, 32, 32), torch.rand(1, 32, 16, 16, 16)]
    d = IterHead(3, [16, 32], 2,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 metrics=[
                     dict(type='Dice', aggregate=None),
                 ])
    outs = d(data, data)
    [print(o.shape) for o in outs]
