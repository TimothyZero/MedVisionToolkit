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

from typing import Union, List
import numpy as np
import math

import torch

from medtk.model.heads_det.dense_heads.anchor_head import AnchorHead
from medtk.model.nnModules import ComponentModule


class SubnetReg(ComponentModule):
    def __init__(self, dim: int, in_channels: int, num_base_anchors=9, feature_size=256,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(SubnetReg, self).__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = self.build_conv(dim, in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = self.build_act(self.act_cfg)

        self.conv2 = self.build_conv(dim, feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = self.build_act(self.act_cfg)

        self.conv3 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act3 = self.build_act(self.act_cfg)

        self.conv4 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = self.build_act(self.act_cfg)

        self.output = self.build_conv(dim, feature_size, num_base_anchors * 2 * dim, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        return out


class SubnetCls(ComponentModule):
    def __init__(self, dim: int, in_channels: int, num_base_anchors=9, num_classes=80, feature_size=256,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(SubnetCls, self).__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.num_classes = num_classes
        self.base_anchors = num_base_anchors

        self.conv1 = self.build_conv(dim, in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = self.build_act(self.act_cfg)

        self.conv2 = self.build_conv(dim, feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = self.build_act(self.act_cfg)

        self.conv3 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act3 = self.build_act(self.act_cfg)

        self.conv4 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = self.build_act(self.act_cfg)

        self.output = self.build_conv(dim, feature_size, num_base_anchors * num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        return out


class RetinaHead(AnchorHead):
    def __init__(self, **kwargs):
        super(RetinaHead, self).__init__(**kwargs)

    def _init_layers(self):
        self.conv_cls = SubnetCls(self.dim,
                                  in_channels=self.in_channels,
                                  num_base_anchors=self.num_anchors,
                                  num_classes=self.cls_out_channels,
                                  feature_size=self.feat_channels,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg,
                                  act_cfg=self.act_cfg)
        self.conv_reg = SubnetReg(self.dim,
                                  in_channels=self.in_channels,
                                  num_base_anchors=self.num_anchors,
                                  feature_size=self.feat_channels,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg,
                                  act_cfg=self.act_cfg)

    def _init_weights(self):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif self.is_norm(self.dim, m):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.conv_cls.output.weight.data.fill_(0)
        self.conv_cls.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.conv_reg.output.weight.data.fill_(0)
        self.conv_reg.output.bias.data.fill_(0)

    def forward_single_level(self, feature):
        cls_out = self.conv_cls(feature)  # (b, nb * classes, (z,) y, x)
        reg_out = self.conv_reg(feature)  # (b, nb * 2 * dim, (z,) y, x)
        return cls_out, reg_out


if __name__ == "__main__":
    from medtk.task import *

    DIM = 2

    net = RetinaHead(
        dim=DIM,
        in_channels=128,
        feat_channels=128,
        num_classes=1,
        level_first=True,
        anchor_generator=AnchorGenerator(
            dim=DIM,
            base_scales=1,
            scales=[0.5, 1, 1.25, 2.5, 5, 8],
            ratios=[1],
            strides=[2, 4]
        ),
        bbox_coder=DeltaBBoxCoder(
            dim=DIM,
            target_stds=[1., 1., 1., 1.]),
        assigner=UniformAssigner(
            neg_ignore_thr=0.8 ** DIM,
            pos_ignore_thr=0.4 ** DIM,
            min_pos_iou=0.3 ** DIM,
            match_low_quality=True,
            num_neg=5000),
        sampler=HNEMSampler(
            pos_fraction=0.25),
        proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.8 ** DIM,
            min_bbox_size=0),
        losses=dict(
            cls=CrossEntropyLoss(use_sigmoid=True),
            reg=SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)),
        metrics=[
            IOU(aggregate='none'),
            Dist(aggregate='none', dist_threshold=5)
        ],
        act_cfg=dict(type='LeakyReLU')
    )
    print(net)

    features = [torch.zeros((3, 128, 24, 24)), torch.zeros((3, 128, 12, 12))]
    gt_labels = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, -1]
    ]).float()
    gt_bboxes = torch.tensor([
        [
            [12, 12, 18, 18],
            [1, 1, 2, 2],
            [1, 1, 4, 4],
        ],
        [
            [12, 18, 18, 22],
            [11, 11, 20, 20],
            [1, 1, 4, 4],
        ],
        [
            [8, 8, 16, 16],
            [4, 4, 8, 8],
            [-1, -1, -1, -1]
        ]
    ]).float()
    loss_dict, batch_bboxes1, (mlvl_label_targets, mlvl_bbox_targets) = net.forward_train(
        features, gt_labels, gt_bboxes)
    print(loss_dict)
    print(batch_bboxes1[0].shape)
    print('end')
