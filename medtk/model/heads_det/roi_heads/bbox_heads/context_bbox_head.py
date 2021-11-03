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
import torch
from torch import nn

from .bbox_head import BBoxHead


class ContextBBoxHead(BBoxHead):
    def __init__(self, roi_scale=2, *args, **kwargs):
        super(ContextBBoxHead, self).__init__(*args, **kwargs)
        self.roi_scale = roi_scale

    def _init_layers(self):
        self.shared_bone = nn.Sequential(
            self.build_conv(self.dim, self.in_channels * 2, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(inplace=True),
        )
        self.roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())
        
        roi_features = self.extractors(feats, rois)
        enlarged_roi_features = self.extractors(feats, rois, roi_scale_factor=self.roi_scale)
        roi_features = torch.cat([roi_features, enlarged_roi_features], dim=1)

        out = self.shared_bone(roi_features)
        out = out.view(-1, self.flatten_features)
        cls_out = self.roi_cls(out)
        reg_out = self.roi_reg(out)
        return cls_out, reg_out
