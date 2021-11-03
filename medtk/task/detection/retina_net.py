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

from collections import OrderedDict

from medtk.model.nnModules import BaseTask


class RetinaNet(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 head=None):
        super(RetinaNet, self).__init__()
        self.dim = dim
        if backbone:
            self.backbone = backbone
        if neck:
            self.neck = neck
        if head:
            self.head = head

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, data_batch: dict, *args, **kwargs):
        self.try_to_info('start')
        img = data_batch['img']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)

        feats = self.extract_feat(img)
        self.try_to_info('extract features')

        loss_dict, batch_bboxes, net_output = self.head.forward_train(feats, gt_labels, gt_bboxes)
        self.try_to_info('loss')
        return loss_dict, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)

        feats = self.extract_feat(img)

        loss_dict, batch_bboxes, net_output = self.head.forward_valid(feats, gt_labels, gt_bboxes)

        metrics = self.metric(data_batch, batch_bboxes)

        metrics_losses = OrderedDict()
        metrics_losses.update(metrics)
        metrics_losses.update(loss_dict)

        return metrics_losses, batch_bboxes, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        img = data_batch['img']

        feats = self.extract_feat(img)
        batch_bboxes, batch_anchor_id, net_output = self.head.forward_infer(feats)

        return batch_bboxes, net_output

    def metric(self, data_batch: dict, prediction):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']
        metrics = self.head.metric(prediction, label)
        return metrics