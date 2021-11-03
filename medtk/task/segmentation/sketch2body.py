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
import torch.nn.functional as F
from collections import OrderedDict

from medtk.model.nnModules import BaseTask


class Sketch2Body(BaseTask):
    def __init__(self,
                 dim,
                 backbone=None,
                 neck=None,
                 seg_head=None,
                 bone_head=None,
                 inpaint_head=None):
        super(Sketch2Body, self).__init__()

        self.dim = dim
        if backbone:
            self.backbone = backbone
        if neck:
            self.neck = neck
        if seg_head:
            self.seg_head = seg_head
        if bone_head:
            self.bone_head = bone_head
        if inpaint_head:
            self.inpaint_head = inpaint_head

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def forward_train(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        seg_label = data_batch['gt_seg']
        bone_label = data_batch['gt_seg_skeleton']
        random_mask = data_batch['cutout_mask']

        assert torch.max(seg_label) <= self.seg_head.out_channels, \
            f'label max = {torch.max(seg_label)} while out_channels = {self.seg_head.out_channels}'

        x_bk = self.backbone(img)
        if self.with_neck:
            x_nk = self.neck(x_bk.copy())
            net_output = self.seg_head(x_bk, x_nk)
            del x_bk
            del x_nk
        else:
            net_output = self.seg_head(None, x_bk)
            del x_bk

        losses = OrderedDict()
        seg_losses = self.loss(data_batch, net_output)

        bone_prediction = self.bone_head(bone_label, random_mask)
        bone_losses = self.bone_head.loss(bone_prediction, bone_label, mask=random_mask)

        seg_refined = self.inpaint_head(bone_label, seg_label, random_mask)
        refined_losses = self.inpaint_head.loss(seg_refined, seg_label, mask=random_mask)

        losses.update(seg_losses)
        losses.update(bone_losses)
        losses.update(refined_losses)

        net_output = [torch.cat([net_output[0], bone_prediction, bone_label, seg_label, seg_refined], dim=1)]

        return losses, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        seg_label = data_batch['gt_seg']
        bone_label = data_batch['gt_seg_skeleton']
        random_mask = data_batch['cutout_mask']

        assert torch.max(seg_label) <= self.seg_head.out_channels, \
            f'label max = {torch.max(seg_label)} while out_channels = {self.seg_head.out_channels}'

        x_bk = self.backbone(img)
        if self.with_neck:
            x_nk = self.neck(x_bk.copy())
            net_output = self.seg_head(x_bk, x_nk)
            del x_bk
            del x_nk
        else:
            net_output = self.seg_head(None, x_bk)
            del x_bk

        seg_losses = self.loss(data_batch, net_output)

        bone_prediction = self.bone_head(bone_label, random_mask)
        # bone_losses = self.bone_head.loss(bone_prediction, bone_label, mask=random_mask)
        bone_losses = self.bone_head.loss(bone_prediction, bone_label, mask=None)

        seg_refined = self.inpaint_head(bone_label, seg_label, random_mask)
        refined_losses = self.inpaint_head.loss(seg_refined, seg_label, mask=random_mask)

        metrics = self.metric(data_batch, seg_refined)

        metrics_losses = OrderedDict()
        metrics_losses.update(seg_losses)
        metrics_losses.update(bone_losses)
        metrics_losses.update(refined_losses)
        metrics_losses.update(metrics)

        return metrics_losses, seg_refined, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        # img = data_batch['img']
        #
        # x_bk = self.backbone(img)
        # if self.with_neck:
        #     x_nk = self.neck(x_bk.copy())
        #     net_output = self.seg_head(x_bk, x_nk)
        # else:
        #     net_output = self.seg_head(None, x_bk)
        #
        # prediction = net_output[0].softmax(dim=1)[:, 1:]
        # return prediction, net_output
        raise NotImplementedError

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

            loss_dict = self.seg_head.loss(prediction, label.long())
            reference = self.seg_head.loss_reference(prediction, label.long())
            # print(loss_dict, reference)
            assert isinstance(loss_dict, dict), "head.loss must return a dict of losses"
            assert isinstance(reference, torch.Tensor), "head.base_loss must return a tensor"

            losses['reference'] = losses.get('reference', 0) + reference * (0.5 ** i) / rescale
            for loss_name, loss_value in loss_dict.items():
                losses[loss_name] = losses.get(loss_name, 0) + loss_value * (0.5 ** i) / rescale
        return losses

    def metric(self, data_batch, prediction) -> dict:
        assert 'gt_seg' in data_batch.keys()
        label = data_batch['gt_seg']
        metrics = self.seg_head.metric(prediction, label)
        return metrics

    # def optimize(self, optimizers, losses):
    #     optimizers['dis_optimizer'].zero_grad()
    #     optimizers['gen_optimizer'].zero_grad()
    #
    #     losses.pop('dis_loss').backward(retain_graph=True)
    #     losses.pop('gen_loss').backward()
    #
    #     optimizers['dis_optimizer'].step()
    #     optimizers['gen_optimizer'].step()
    #
    #     loss, log_vars = self.parse_losses(losses)
    #
    #     loss.backward()
    #     optimizers['segmentation'].step()
    #     return log_vars