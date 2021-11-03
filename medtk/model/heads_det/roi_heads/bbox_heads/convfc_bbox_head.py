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
from torch import nn

from .bbox_head import BBoxHead


class ConvFCBBoxHead(BBoxHead):
    def __init__(self,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.shared_convs = nn.Sequential(
            self.build_conv(self.dim, self.in_channels, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(inplace=True),
        )
        self.roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

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
        # self.roi_cls.weight.data.fill_(0)
        # self.roi_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.roi_reg.weight.data.fill_(0)
        # self.roi_reg.bias.data.fill_(0)
        #
        # # normal_init(self.roi_cls, std=0.01)
        # # normal_init(self.roi_reg, std=0.01)

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())

        roi_features = self.extractors(feats, rois)

        out = self.shared_convs(roi_features)
        out = out.view(-1, self.flatten_features)
        cls_out = self.roi_cls(out)
        reg_out = self.roi_reg(out)
        return cls_out, reg_out