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


class GradientSegHead(nn.Module):
    def __init__(self, gra_channels, in_channels, num_classes, loss_cls):
        super(GradientSegHead, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'must be a list/tuple'
        self.gra_channels = gra_channels
        self.in_channels = in_channels
        self.out_channels = num_classes + 1
        self.criterion = loss_cls
        self.base_criterion = nn.CrossEntropyLoss()
        self.layers = None
        self.gra_layers = None
        self.init_layers()
    
    def init_layers(self):
        layers = []
        for in_channel in self.in_channels:
            layers.append(self._make_single_out(in_channel, self.out_channels, 1, 0))
        self.layers = nn.ModuleList(layers)

        layers = []
        for in_channel in self.in_channels:
            layers.append(self._make_single_out(self.gra_channels, in_channel, 3, 1))
        self.gra_layers = nn.ModuleList(layers)
    
    def _make_single_out(self, in_channels, out_channels, kernel_size, padding):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    
    def forward(self, x, gradient):
        assert len(self.in_channels) == len(x), 'backbone/neck outs not match head ins'
        outs = []
        for i in range(len(self.in_channels)):
            gra = F.avg_pool2d(gradient, 2 ** i)  # do not use max pool
            # print(gra.shape, x[i].shape)
            out = self.gra_layers[i](gra)
            out = self.layers[i](x[i] + out)
            outs.append(out)
        return outs
    
    def loss(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
    
    def loss_reference(self, prediction, ground_truth):
        return self.base_criterion(prediction, ground_truth)