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
import torch.functional as F
from collections import OrderedDict

from medtk.model.nnModules import BaseTask


class EdgeConnectSeg(BaseTask):
    def __init__(self,
                 dim,
                 backbone=None,
                 neck=None,
                 head=None,
                 tree_head=None):
        super(EdgeConnectSeg, self).__init__()

        self.dim = dim
        if backbone:
            self.backbone = backbone
        if neck:
            self.neck = neck
        if head:
            self.head = head
        if tree_head:
            self.tree_head = tree_head

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def forward_train(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        label = data_batch['gt_seg']  # .long()
        assert 'gt_seg' in data_batch.keys()
        assert torch.max(label) <= self.head.out_channels, \
            f'label max = {torch.max(label)} while out_channels = {self.head.out_channels}'

        x_bk = self.backbone(img)
        if self.with_neck:
            x_nk = self.neck(x_bk.copy())
            net_output = self.head(x_bk, x_nk)
            del x_bk
            del x_nk
        else:
            net_output = self.head(None, x_bk)
            del x_bk

        losses = self.loss(data_batch, net_output)

        images, edges, masks = net_output[0], data_batch['gt_tree'], data_batch['cutout_mask']
        outputs, gen_loss, dis_loss, logs = self.tree_head.process(images, edges, masks)

        # losses = self.edge_loss(data_batch, outputs)
        losses['gen_loss'] = gen_loss
        losses['gen_loss'] = dis_loss

        return losses, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        label = data_batch['gt_seg']  # .long()

        x_bk = self.backbone(img)
        if self.with_neck:
            x_nk = self.neck(x_bk.copy())
            net_output = self.head(x_bk, x_nk)
        else:
            net_output = self.head(None, x_bk)
            del x_bk

        prediction = net_output[0]

        losses = self.loss(data_batch, net_output[[0]])

        metrics = self.metric(data_batch, prediction)

        metrics_losses = OrderedDict()
        metrics_losses.update(metrics)
        metrics_losses.update(losses)

        return metrics_losses, prediction, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        img = data_batch['img']

        x_bk = self.backbone(img)
        if self.with_neck:
            x_nk = self.neck(x_bk.copy())
            net_output = self.head(x_bk, x_nk)
        else:
            net_output = self.head(None, x_bk)

        prediction = net_output[0].softmax(dim=1)[:, 1:]
        return prediction, net_output

    def loss(self, data_batch: dict, net_output: list) -> dict:
        assert 'gt_seg' in data_batch.keys()
        label = data_batch['gt_seg']

        rescale = 2 * (1 - 0.5 ** len(net_output))
        # print(rescale, len(net_output))
        losses = OrderedDict()
        for i in range(len(net_output)):
            prediction = net_output[i]
            if prediction.shape[2:] != label.shape[1:]:
                prediction = F.interpolate(prediction, scale_factor=2 ** i)

            loss_dict = self.head.loss(prediction, label.long())
            reference = self.head.loss_reference(prediction, label.long())
            # print(loss_dict, reference)
            assert isinstance(loss_dict, dict), "head.loss must return a dict of losses"
            assert isinstance(reference, torch.Tensor), "head.base_loss must return a tensor"

            losses['reference'] = losses.get('reference', 0) + reference * (0.5 ** i) / rescale
            for loss_name, loss_value in loss_dict.items():
                losses[loss_name] = losses.get(loss_name, 0) + loss_value * (0.5 ** i) / rescale
        return losses

    def edge_loss(self, data_batch: dict, prediction: torch.Tensor) -> dict:
        assert 'gt_edge' in data_batch.keys()
        label = data_batch['gt_edge']
        loss_dict = self.head.loss(prediction, label.long())
        reference = self.head.loss_reference(prediction, label.long())
        loss_dict['reference'] = reference
        return loss_dict

    def metric(self, data_batch, prediction) -> dict:
        assert 'gt_seg' in data_batch.keys()
        label = data_batch['gt_seg']
        metrics = self.head.metric(prediction, label)
        return metrics

    def optimize(self, optimizers, losses):
        optimizers['dis_optimizer'].zero_grad()
        optimizers['gen_optimizer'].zero_grad()

        losses.pop('dis_loss').backward(retain_graph=True)
        losses.pop('gen_loss').backward()

        optimizers['dis_optimizer'].step()
        optimizers['gen_optimizer'].step()

        loss, log_vars = self.parse_losses(losses)

        loss.backward()
        optimizers['segmentation'].step()
        return log_vars