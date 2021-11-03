#  Copyright (c) 2021. The Medical Image Computing (MIC) Lab, 陶豪毅
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

# -*- coding:utf-8 -*-
from typing import Union, List
from torch import nn
import torch
import numpy as np
import math
import warnings

from medtk.model.nd import AdaptiveAvgPoolNd

from .bbox_head import BBoxHead


class ConvBBoxHead(BBoxHead):
    def __init__(self, *args, **kwargs):
        super(ConvBBoxHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.shared_convs = nn.Sequential(
            self.build_conv(self.dim, self.in_channels, self.feature_channels, kernel_size=3),
            self.build_norm(self.dim, self.feature_channels, norm_cfg=dict(type='GroupNorm', num_groups=32, eps=1e-05, affine=True),),
            self.build_act(inplace=True),
            self.build_conv(self.dim, self.feature_channels, self.feature_channels, kernel_size=3),
            self.build_norm(self.dim, self.feature_channels, norm_cfg=dict(type='GroupNorm', num_groups=32, eps=1e-05, affine=True),),
            self.build_act(inplace=True),
            self.build_conv(self.dim, self.feature_channels, self.feature_channels, kernel_size=3),
            self.build_norm(self.dim, self.feature_channels, norm_cfg=dict(type='GroupNorm', num_groups=32, eps=1e-05, affine=True),),
            self.build_act(inplace=True),
            AdaptiveAvgPoolNd(self.dim)(1),
        )

        self.shared_fcs = nn.Sequential(
            nn.Linear(self.feature_channels, self.fc_channels),
            self.build_act(inplace=True),
        )
        self.roi_cls = nn.Linear(self.fc_channels, self.num_classes + 1)
        self.roi_reg = nn.Linear(self.fc_channels, 2 * self.dim)

    def _init_weights(self):
        pass

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())

        roi_features = self.extractors(feats, rois)

        out = self.shared_convs(roi_features)
        if self.dim == 2:
            out = out.view(-1, self.flatten_features)
        else:
            out = out.view(-1, self.feature_channels)
        out = self.shared_fcs(out)
        cls_out = self.roi_cls(out)
        reg_out = self.roi_reg(out)
        return cls_out, reg_out
