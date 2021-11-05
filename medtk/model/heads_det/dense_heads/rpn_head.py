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

from typing import Union, List
import torch.nn as nn
import math
import numpy as np

from medtk.model.heads_det.dense_heads.anchor_head import AnchorHead


class RPNHead(AnchorHead):

    def __init__(self, **kwargs):
        kwargs['num_classes'] = 1
        super(RPNHead, self).__init__(**kwargs)

    def _init_layers(self):
        self.rpn_bone = nn.Sequential(
            self.build_conv(self.dim, self.in_channels, self.feat_channels, kernel_size=3, padding=1),
            self.build_act(inplace=True),
            self.build_conv(self.dim, self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.build_act(inplace=True),
            self.build_conv(self.dim, self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.build_act(inplace=True),
            self.build_conv(self.dim, self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
        )

        self.act = self.build_act(inplace=True)
        self.rpn_cls = self.build_conv(self.dim, self.feat_channels,
                                       self.num_anchors * self.num_classes, kernel_size=1)
        self.rpn_reg = self.build_conv(self.dim, self.feat_channels,
                                       self.num_anchors * 2 * self.dim, kernel_size=1)

    def _init_weights(self):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif self.is_norm(self.dim, m):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.rpn_cls.weight.data.fill_(0)
        self.rpn_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.rpn_reg.weight.data.fill_(0)
        self.rpn_reg.bias.data.fill_(0)

    def forward_single_level(self, single_level_feature):
        feature = self.rpn_bone(single_level_feature)
        feature = self.act(feature)
        cls_out = self.rpn_cls(feature)  # (b, nb * 1, (z,) y, x)
        reg_out = self.rpn_reg(feature)  # (b, nb * 2 * dim, (z,) y, x)
        return cls_out, reg_out