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

from typing import List, Union
import torch
from torch import nn

from medtk.model.nd import ReflectionPad3d
from medtk.model.nnModules import ComponentModule
from ..losses import CrossEntropyLoss


class EdgeGenerator(nn.Module):
    def __init__(self, base_width=64, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            ReflectionPad3d(3),
            spectral_norm(nn.Conv3d(in_channels=1, out_channels=base_width, kernel_size=7, padding=0),
                          use_spectral_norm),
            nn.InstanceNorm3d(base_width, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(
                nn.Conv3d(in_channels=base_width, out_channels=base_width * 2, kernel_size=4, stride=2, padding=1),
                use_spectral_norm),
            nn.InstanceNorm3d(base_width * 2, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(
                nn.Conv3d(in_channels=base_width * 2, out_channels=base_width * 4, kernel_size=4, stride=2, padding=1),
                use_spectral_norm),
            nn.InstanceNorm3d(base_width * 4, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(base_width * 4, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(
                nn.ConvTranspose3d(in_channels=base_width * 4, out_channels=base_width * 2, kernel_size=4, stride=2,
                                   padding=1),
                use_spectral_norm),
            nn.InstanceNorm3d(base_width * 2, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(
                nn.ConvTranspose3d(in_channels=base_width * 2, out_channels=base_width, kernel_size=4, stride=2,
                                   padding=1),
                use_spectral_norm),
            nn.InstanceNorm3d(base_width, track_running_stats=False),
            nn.ReLU(True),

            ReflectionPad3d(3),
            nn.Conv3d(in_channels=base_width, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm3d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        # x = torch.sigmoid(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ReflectionPad3d(dilation),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
            nn.ReLU(True),

            ReflectionPad3d(1),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class BoneSegHead(ComponentModule):
    def __init__(self,
                 dim: int,
                 in_channels: List[int],
                 num_classes: int,
                 loss_cls: object,
                 metrics: List[object]):
        super(BoneSegHead, self).__init__()
        assert isinstance(in_channels, list), 'must be a list'
        assert isinstance(in_channels, list), 'must be a list'
        metrics = [metrics] if isinstance(metrics, dict) else metrics

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = num_classes + 1
        self.criterion = loss_cls
        self.base_criterion = CrossEntropyLoss()
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self.layers = None
        self.model = EdgeGenerator(use_spectral_norm=False)

    def forward(self, bone_label, random_mask):
        outs = self.model(bone_label * (1 - random_mask))
        return outs

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output, ground_truth)
            metrics.update(one_metric)
        return metrics

    def loss(self, net_output, ground_truth, mask=None):
        # print(torch.max(net_output), torch.max(ground_truth))
        losses = self.criterion(net_output, ground_truth.long(), weight=mask)
        # print(losses)
        if isinstance(losses, torch.Tensor):
            losses = {'bone_loss': losses}
        return losses

    def loss_reference(self, net_output, ground_truth, mask=None):
        return self.base_criterion(net_output, ground_truth, weight=mask)
