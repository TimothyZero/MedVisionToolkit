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


class MultiContextBBoxHead(BBoxHead):
    def __init__(self, roi_scale=2, ens_method='mean', *args, **kwargs):
        super(MultiContextBBoxHead, self).__init__(*args, **kwargs)
        assert ens_method in ['mean', 'max']
        self.roi_scale = roi_scale
        self.ensemble_method = ens_method
        self.sim = nn.CosineSimilarity(dim=1)

    def _init_layers(self):
        self.dr1_shared_bone = nn.Sequential(
            self.build_conv(self.dim, self.in_channels * 2, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(self.act_cfg),
        )
        self.dr1_roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.dr1_roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

        self.dr2_shared_bone = nn.Sequential(
            self.build_conv(self.dim, self.in_channels * 2, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(self.act_cfg),
        )
        self.dr2_roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.dr2_roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

        self.dr3_shared_bone = nn.Sequential(
            self.build_conv(self.dim, self.in_channels * 2, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(self.act_cfg),
        )
        self.dr3_roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.dr3_roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())
        
        roi_features = self.extractors(feats, rois)

        enlarged_roi_features = self.extractors(feats, rois, roi_scale_factor=self.roi_scale)
        roi_features = torch.cat([roi_features, enlarged_roi_features], dim=1)

        dr1_out = self.dr1_shared_bone(roi_features)
        dr1_out = dr1_out.view(-1, self.flatten_features)
        dr1_cls_out = self.dr1_roi_cls(dr1_out)
        dr1_reg_out = self.dr1_roi_reg(dr1_out)

        dr2_out = self.dr2_shared_bone(roi_features)
        dr2_out = dr2_out.view(-1, self.flatten_features)
        dr2_cls_out = self.dr2_roi_cls(dr2_out)
        dr2_reg_out = self.dr2_roi_reg(dr2_out)

        dr3_out = self.dr3_shared_bone(roi_features)
        dr3_out = dr3_out.view(-1, self.flatten_features)
        dr3_cls_out = self.dr3_roi_cls(dr3_out)
        dr3_reg_out = self.dr3_roi_reg(dr3_out)

        if self.ensemble_method == 'mean':
            cls_out = (dr1_cls_out + dr2_cls_out + dr3_cls_out) / 3
        elif self.ensemble_method == 'max':
            cls_out = torch.maximum(torch.maximum(dr1_cls_out, dr2_cls_out), dr3_cls_out)
        else:
            raise NotImplementedError

        reg_out = (dr1_reg_out + dr2_reg_out + dr3_reg_out) / 3

        sim = 1 / 3 * (self.sim(dr1_out, dr2_out) + self.sim(dr1_out, dr3_out) + self.sim(dr3_out, dr2_out))
        self.sim_loss = 0.5 * sim

        return cls_out, reg_out

    def get_losses(self, cls_out, reg_out, rois_labels, rois_deltas):
        losses = super().get_losses(cls_out, reg_out, rois_labels, rois_deltas)
        losses['sim_loss'] = self.sim_loss
        return losses