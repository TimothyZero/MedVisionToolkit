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

from torch import nn

from medtk.model.nd import ConvNd, BatchNormNd, AdaptiveAvgPoolNd
from medtk.model.nnModules import ComponentModule


class BaseClsHead(ComponentModule):
    def __init__(self, dim, in_channels, num_classes, loss_cls, metrics):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)), 'must be a list/tuple'

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = num_classes
        fcs = []
        for in_channel in in_channels:
            fcs.append(nn.Linear(in_channel, self.out_channels))
        self.fcs = nn.ModuleList(fcs)
        self.avgpool = AdaptiveAvgPoolNd(self.dim)((1,) * self.dim)
        
        self.criterion = loss_cls
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self.base_criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        assert len(self.in_channels) == len(x), 'backbone/neck outs not match head ins'
        outs = []
        for i in range(len(self.in_channels)):
            x = self.avgpool(x[i]).flatten(1)
            # self.try_to_info("x", x.shape)
            outs.append(self.fcs[i](x))
        return outs

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output.cpu(), ground_truth.cpu())
            metrics.update(one_metric)
        return metrics
    
    def loss(self, prediction, ground_truth):
        return {'loss': self.criterion(prediction, ground_truth)}
    
    def loss_reference(self, prediction, ground_truth):
        return self.base_criterion(prediction, ground_truth)
