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
import torch.nn as nn
import numpy as np
import math

from medtk.model.heads_det.dense_heads.anchor_head import AnchorHead


class DeepLungHead(AnchorHead):

    def __init__(self, **kwargs):
        super(DeepLungHead, self).__init__(**kwargs)
        self._init_weights()

    def _init_layers(self):
        self.output = nn.Sequential(nn.Conv3d(self.in_channels, self.feat_channels, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(self.feat_channels, 7 * self.num_anchors, kernel_size=1))

    def _init_weights(self):
        pass
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         n = np.prod(m.kernel_size) * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif self.is_norm(self.dim, m):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #
        # prior = 0.01
        # self.rpn_cls.weight.data.fill_(0)
        # self.rpn_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.rpn_reg.weight.data.fill_(0)
        # self.rpn_reg.bias.data.fill_(0)

    def forward_single_level(self, single_level_feature):
        out = self.output(single_level_feature)
        cls_out = out[:, :self.anchor_generator.num_base_anchors, ...]
        reg_out = out[:, self.anchor_generator.num_base_anchors:, ...]
        return cls_out, reg_out