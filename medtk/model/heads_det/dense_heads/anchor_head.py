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
from typing import List, Union
import warnings
import torch
from torch import nn
import numpy as np
import math

from medtk.model.heads_det.dense_heads.base_dense_head import BaseDenseHead

from medtk.ops import nmsNd_pytorch, softnmsNd_pytorch
from medtk.utils import multi_apply

try:
    from medvision.ops import nms_nd as nmsNd_cuda
except ImportError:
    warnings.warn('no nms cuda!')
    from medtk.ops import nmsNd_pytorch as nmsNd_cuda


class AnchorHead(BaseDenseHead):
    def __init__(self,
                 dim: int,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 anchor_generator=None,
                 assigner=None,
                 sampler=None,
                 bbox_coder=None,
                 proposal: dict = None,
                 nms: dict = None,
                 losses: dict = None,
                 metrics: Union[List[dict], List[object]] = None,
                 level_first: bool = False,
                 conv_cfg: dict = None,
                 norm_cfg: dict = None,
                 act_cfg: dict = None):
        super(AnchorHead, self).__init__(conv_cfg, norm_cfg, act_cfg)

        self.dim = dim
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.level_first = level_first
        self.use_sigmoid_cls = losses['cls'].get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.anchor_generator = anchor_generator
        self.anchor_assigner = assigner
        self.anchor_sampler = sampler
        self.num_anchors = self.anchor_generator.num_base_anchors

        self.bbox_coder = bbox_coder

        self.nms = nms
        self.proposal_cfg = proposal

        self.base_criterion = nn.CrossEntropyLoss()
        self.criterion_cls = losses['cls']
        self.criterion_reg = losses['reg']
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 2 * self.dim, 1)

    def _init_weights(self):
        """Initialize weights of the head."""
        prior = 0.01
        self.conv_cls.weight.data.fill_(0)
        self.conv_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.conv_reg.weight.data.fill_(0)
        self.conv_reg.bias.data.fill_(0)

    def forward_single_level(self, feature):
        """

        Args:
            feature: single_level_feature, (b, c, **shape)

        Returns:

        """
        cls_out = self.conv_cls(feature)  # (b, nb * class, z, y, x)
        reg_out = self.conv_reg(feature)  # (b, nb * 2 * dim, z, y, x)
        return cls_out, reg_out

    def forward(self, features):
        """Forward features from the upstream network.

        Args:
            features (list[Tensor]): Features from the upstream network, each is
                a 4D/5D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D/5D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D/5D-tensor, the channels number \
                    is num_anchors * 2 * dim.
        """
        mlvl_cls_out, mlvl_reg_out = multi_apply(self.forward_single_level, features)
        return mlvl_cls_out, mlvl_reg_out

    def flatten_forward(self, mlvl_cls_out, mlvl_reg_out, mlvl_anchors, mlvl_valid):
        cls_out_mat, reg_out_mat, anchor_mat, valid_mat = multi_apply(
            self._flatten_forward_single,
            mlvl_cls_out,
            mlvl_reg_out,
            mlvl_anchors,
            mlvl_valid
        )
        # [batch, total_anchors_of_mlvl, classes or 2dim]
        cls_out_mat = torch.cat(cls_out_mat, dim=1)
        reg_out_mat = torch.cat(reg_out_mat, dim=1)
        anchor_mat = torch.cat(anchor_mat, dim=1)
        valid_mat = torch.cat(valid_mat, dim=1)
        return cls_out_mat, reg_out_mat, anchor_mat, valid_mat

    def _flatten_forward_single(self, cls_out, reg_out, anchors, anchors_valid):
        batch_size = cls_out.shape[0]
        axes = [0] + list(range(2, self.dim + 2)) + [1]
        cls_out_flat = cls_out.permute(*axes).contiguous().view(batch_size, -1, self.cls_out_channels)  # (b, Num, cls)
        reg_out_flat = reg_out.permute(*axes).contiguous().view(batch_size, -1, 2 * self.dim)  # (b, Num, 2dim)
        anchors_flat = anchors.permute(*axes).contiguous().view(batch_size, -1, 2 * self.dim)  # (b, Num, 2dim)
        valid_flat = anchors_valid.permute(*axes).contiguous().view(batch_size, -1, 2 * self.dim)  # (b, Num, 2dim)
        return cls_out_flat, reg_out_flat, anchors_flat, valid_flat

    def flatten_backward(self,
                         label_targets_mat,
                         bbox_targets_mat,
                         num_level_anchors,
                         feature_shapes):

        mlvl_label_targets_flat = label_targets_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_bbox_targets_flat = bbox_targets_mat.split_with_sizes(num_level_anchors, dim=1)

        label_targets, bbox_targets = multi_apply(
            self._flatten_backward_single,
            mlvl_label_targets_flat,
            mlvl_bbox_targets_flat,
            feature_shapes
        )
        return label_targets, bbox_targets

    def _flatten_backward_single(self,
                                 label_targets,
                                 bbox_targets,
                                 feature_shape):
        batch_size = label_targets.shape[0]
        axes = [0, self.dim + 1] + list(range(1, self.dim + 1))
        label_targets = label_targets.view(batch_size, *feature_shape, -1).permute(*axes).contiguous()
        bbox_targets = bbox_targets.view(batch_size, *feature_shape, -1).permute(*axes).contiguous()

        return label_targets, bbox_targets

    def get_target(self,
                   batch_gt_labels,
                   batch_gt_bboxes,
                   anchor_mat,
                   valid_mat):
        batch_label_targets, batch_bbox_targets = multi_apply(
            self._target_single_image,
            batch_gt_labels,
            batch_gt_bboxes,
            anchor_mat,
            valid_mat
        )
        batch_label_targets = torch.stack(batch_label_targets, dim=0)
        batch_bbox_targets = torch.stack(batch_bbox_targets, dim=0)
        return batch_label_targets, batch_bbox_targets

    def _target_single_image(self, gt_labels, gt_bboxes, concat_anchors, concat_valid):
        gt_bboxes = gt_bboxes[gt_labels > 0]
        gt_labels = gt_labels[gt_labels > 0].long()

        label_targets, bboxes_targets = self.anchor_assigner.assign(concat_anchors, gt_bboxes, gt_labels)

        label_targets[~ concat_valid[:, 0]] = -1
        bboxes_targets[~ concat_valid] = -1

        return label_targets, bboxes_targets

    def get_samples(self,
                    cls_out_mat,
                    reg_out_mat,
                    label_targets_mat,
                    bbox_targets_mat,
                    anchor_mat,
                    valid_mat,
                    num_level_anchors):

        mlvl_cls_out_flat = cls_out_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_reg_out_flat = reg_out_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_label_targets_flat = label_targets_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_bbox_targets_flat = bbox_targets_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_anchor_flat = anchor_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_valid_flat = valid_mat.split_with_sizes(num_level_anchors, dim=1)

        mlvl_label_targets_flat_sampled, mlvl_bbox_targets_flat_sampled = multi_apply(
            self._samples_single_level,
            mlvl_cls_out_flat,
            mlvl_reg_out_flat,
            mlvl_label_targets_flat,
            mlvl_bbox_targets_flat,
            mlvl_anchor_flat,
            mlvl_valid_flat,
        )

        label_targets_mat_sampled = torch.cat(mlvl_label_targets_flat_sampled, dim=1)
        bbox_targets_mat_sampled = torch.cat(mlvl_bbox_targets_flat_sampled, dim=1)
        return label_targets_mat_sampled, bbox_targets_mat_sampled

    def _samples_single_level(self,
                              cls_out_flat,
                              reg_out_flat,
                              label_targets_flat,
                              bbox_targets_flat,
                              anchor,
                              valid):
        not_ignored = label_targets_flat >= 0
        with torch.no_grad():
            losses = self.criterion_cls(
                cls_out_flat[not_ignored],
                label_targets_flat[not_ignored],
                reduction_override='none').squeeze(-1)

        sampled_pos_indices, sampled_neg_indices, _, _, _ = self.anchor_sampler.sample(
            None,
            label_targets_flat[not_ignored],
            bbox_targets_flat[not_ignored],
            gt_labels=None,
            gt_bboxes=None,
            weight=losses)

        not_ignored_ind = torch.nonzero(not_ignored, as_tuple=True)
        sampled_pos_ind = [i[sampled_pos_indices] for i in not_ignored_ind]
        sampled_neg_ind = [i[sampled_neg_indices] for i in not_ignored_ind]

        label_targets_sampled = label_targets_flat.clone()
        label_targets_sampled[not_ignored] = -1
        label_targets_sampled[sampled_pos_ind] = label_targets_flat[not_ignored][sampled_pos_indices]
        label_targets_sampled[sampled_neg_ind] = label_targets_flat[not_ignored][sampled_neg_indices]

        bbox_targets_sampled = bbox_targets_flat.clone()
        bbox_targets_sampled[not_ignored] = -1
        bbox_targets_sampled[sampled_pos_ind] = bbox_targets_flat[not_ignored][sampled_pos_indices]
        bbox_targets_sampled[sampled_neg_ind] = bbox_targets_flat[not_ignored][sampled_neg_indices]

        return label_targets_sampled, bbox_targets_sampled

    def get_samples_losses(self,
                           cls_out_mat,
                           reg_out_mat,
                           label_targets_mat_sampled,
                           bbox_targets_mat_sampled,
                           anchor_mat,
                           valid_mat,
                           num_level_anchors):

        mlvl_cls_out_flat = cls_out_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_reg_out_flat = reg_out_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_label_targets_flat = label_targets_mat_sampled.split_with_sizes(num_level_anchors, dim=1)
        mlvl_bbox_targets_flat = bbox_targets_mat_sampled.split_with_sizes(num_level_anchors, dim=1)
        mlvl_anchor_flat = anchor_mat.split_with_sizes(num_level_anchors, dim=1)
        mlvl_valid_flat = valid_mat.split_with_sizes(num_level_anchors, dim=1)

        losses_cls, losses_bbox, tpr, tnr, num_pos, num_neg = multi_apply(
            self._samples_losses_single_level,
            mlvl_cls_out_flat,
            mlvl_reg_out_flat,
            mlvl_label_targets_flat,
            mlvl_bbox_targets_flat,
            mlvl_anchor_flat,
            mlvl_valid_flat
        )
        return dict(reg_loss=losses_bbox,
                    cls_loss=losses_cls,
                    tnr=tnr, tpr=tpr,
                    num_pos=num_pos, num_neg=num_neg)

    def _samples_losses_single_level(self,
                                     cls_out_flat,
                                     reg_out_flat,
                                     label_targets_sampled,
                                     bbox_targets_sampled,
                                     anchors,
                                     valid):
        pos_indices_sampled = label_targets_sampled > 0
        neg_indices_sampled = label_targets_sampled == 0

        num_pos = pos_indices_sampled.sum()
        num_neg = neg_indices_sampled.sum()

        bbox_targets_sampled[pos_indices_sampled] = self.bbox_coder.encode(anchors[pos_indices_sampled],
                                                                           bbox_targets_sampled[pos_indices_sampled])

        if num_pos > 0:
            sampled_pos_cls = cls_out_flat[pos_indices_sampled]
            sampled_pos_label = label_targets_sampled[pos_indices_sampled]

            pos_cls_loss = self.criterion_cls(
                sampled_pos_cls,
                sampled_pos_label
            )

            if getattr(self.criterion_reg, 'decode_bbox', False):
                pos_reg_loss = self.criterion_reg(
                    self.bbox_coder.decode(anchors[pos_indices_sampled],
                                           reg_out_flat[pos_indices_sampled]),
                    self.bbox_coder.decode(anchors[pos_indices_sampled],
                                           bbox_targets_sampled[pos_indices_sampled]),
                    reduction_override='none'
                )
            else:
                pos_reg_loss = self.criterion_reg(
                    reg_out_flat[pos_indices_sampled],
                    bbox_targets_sampled[pos_indices_sampled],
                    reduction_override='none'
                )

            pos_reg_loss = torch.mean(pos_reg_loss, dim=0).sum()

            tpr = 100.0 * (sampled_pos_cls > 0).sum() / len(sampled_pos_cls)
        else:
            pos_cls_loss = torch.tensor(0.0, requires_grad=True).to(cls_out_flat.device)
            pos_reg_loss = torch.tensor(0.0, requires_grad=True).to(cls_out_flat.device)
            tpr = 100.0 * torch.tensor(0.0).to(cls_out_flat.device)

        if num_neg > 0:
            sampled_neg_cls = cls_out_flat[neg_indices_sampled]
            sampled_neg_label = label_targets_sampled[neg_indices_sampled]
            neg_cls_loss = self.criterion_cls(
                sampled_neg_cls,
                sampled_neg_label)
            tnr = 100.0 * (sampled_neg_cls < 0).sum() / len(sampled_neg_cls)
        else:
            neg_cls_loss = torch.tensor(0.0, requires_grad=True).to(cls_out_flat.device)
            tnr = 100.0 * torch.tensor(0.0).to(cls_out_flat.device)

        loss_cls = 0.5 * neg_cls_loss + 0.5 * pos_cls_loss
        loss_bbox = pos_reg_loss

        return loss_cls, loss_bbox, tpr, tnr, num_pos, num_neg

    def get_bboxes(self, cls_out_mat, reg_out_mat, anchor_mat, valid_mat):
        batch_bboxes, batch_anchor_id = multi_apply(
            self._bboxes_single_image,
            cls_out_mat,
            reg_out_mat,
            anchor_mat,
            valid_mat
        )
        return torch.stack(batch_bboxes), torch.stack(batch_anchor_id)

    def _bboxes_single_image(self, cls_out_flat, reg_out_flat, concat_level_anchors, concat_valid):
        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            if cls_out_flat.is_cuda:
                nms_fun = nmsNd_cuda
            else:
                nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        nms_pre = self.nms['nms_pre']
        min_bbox_size = self.nms['min_bbox_size']
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        self.try_to_info("postprocess reg_out_flat, anchors_flat",
                         cls_out_flat.shape, reg_out_flat.shape, concat_level_anchors.shape)

        cls_out_flat = cls_out_flat.sigmoid()
        cls_out_flat[~ concat_valid[:, 0]] = 0

        bboxes = self.bbox_coder.decode(concat_level_anchors, reg_out_flat)
        self.try_to_info('bboxes', bboxes)

        results = np.ones((0, self.dim * 2 + 2)) * -1
        anchor_ids = np.ones((0, ),  dtype=np.int) * -1
        for i in range(cls_out_flat.shape[1]):  # multi classes
            scores = torch.squeeze(cls_out_flat[:, i])
            labels = torch.ones_like(scores) * (i + 1)

            _, order = torch.sort(scores, descending=True)
            order = order[:nms_pre]
            scores = scores[order]
            bboxes = bboxes[order]
            labels = labels[order]

            keep, _ = nms_fun(torch.cat([bboxes, scores.unsqueeze(-1)], dim=1), iou_threshold)
            scores, labels, bboxes, anchor_id = scores[keep], labels[keep], bboxes[keep], order[keep]

            high_score_indices = np.where(scores.cpu() > score_threshold)[0]
            for j in range(len(high_score_indices)):
                bbox = bboxes[high_score_indices[j]]
                bbox = bbox.detach().cpu().numpy()
                # if np.min(bbox[self.dim:2 * self.dim] - bbox[:self.dim]) < min_bbox_size:
                #     continue
                label = int(labels[high_score_indices[j]].item())
                score = scores[high_score_indices[j]].item()
                this_anchor_id = anchor_id[high_score_indices[j]].item()
                self.try_to_info("postprocess", [*bbox, label, score])
                results = np.concatenate([
                    results,
                    np.array([[*bbox, label, score]])
                ], axis=0)
                anchor_ids = np.concatenate([
                    anchor_ids,
                    np.array([this_anchor_id],  dtype=np.int)
                ], axis=0)

        results = torch.from_numpy(results)
        anchor_ids = torch.from_numpy(anchor_ids)
        _, order = torch.sort(results[:, -1], descending=True)
        results = results[order[:max_per_img]]
        anchor_ids = anchor_ids[order[:max_per_img]]
        padded_results = cls_out_flat.new_ones((max_per_img, self.dim * 2 + 2)) * -1
        padded_results[:results.shape[0]] = results
        padded_anchor_ids = cls_out_flat.new_ones((max_per_img,), dtype=torch.int64) * -1
        padded_anchor_ids[:anchor_ids.shape[0]] = anchor_ids
        return padded_results, padded_anchor_ids

    def has_proposal(self):
        return self.get('proposal_cfg', None) is not None

    def get_proposals(self, cls_out_mat, reg_out_mat, anchor_mat, valid_mat):
        if not self.has_proposal():
            return None

        batch_proposals, batch_anchor_id = multi_apply(
            self._proposal_single_image,
            cls_out_mat,
            reg_out_mat,
            anchor_mat,
            valid_mat
        )
        return batch_proposals, batch_anchor_id

    def _proposal_single_image(self, cls_out_flat, reg_out_flat, concat_level_anchors, concat_valid):
        pre_nms_limit = self.proposal_cfg['nms_pre']
        max_output_num = self.proposal_cfg['max_num']
        nms_threshold = self.proposal_cfg['nms_thr']

        # proposals = torch.ones((max_output_num, 2 * self.dim), device=cls_out_flat.device) * -1

        self.try_to_info("Proposal")

        cls_out_flat_scores = cls_out_flat.squeeze(-1).sigmoid()  # squeeze class channel
        cls_out_flat_scores[~ concat_valid[:, 0]] = 0

        scores, order = torch.sort(cls_out_flat_scores, descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = reg_out_flat[order, :]
        cur_anchors = concat_level_anchors[order, :].clone()

        boxes = self.bbox_coder.decode(cur_anchors, deltas)

        dets = torch.cat([boxes, scores.unsqueeze(-1)], dim=1)
        if cls_out_flat.is_cuda:
            keep, _ = nmsNd_cuda(dets, nms_threshold)
        else:
            keep, _ = nmsNd_pytorch(dets, nms_threshold)
        keep = keep[:max_output_num]
        proposals = boxes[keep]
        anchor_id = order[keep]
        # keep = keep[:max_output_num]
        # proposals = boxes[keep].detach().cpu().numpy()
        # scores = scores[keep].detach().cpu().numpy()
        #
        # proposals = np.concatenate([proposals, np.ones_like(scores[:, None]), scores[:, None]], axis=1)

        # self.info("proposal", proposals.shape)
        return proposals, anchor_id

    def forward_train(self,
                      feats: List[torch.Tensor],
                      batch_gt_labels: torch.Tensor,
                      batch_gt_bboxes: torch.Tensor):
        self.try_to_info('start anchor based')
        mlvl_cls_out, mlvl_reg_out = self.forward(feats)

        feature_shapes = [f.size()[2:] for f in mlvl_cls_out]
        batch_size = mlvl_cls_out[0].shape[0]
        device = mlvl_cls_out[0].device

        # e.g. 2D image, batch size = 3, level = 2, nb = num of base anchor
        # mlvl_anchors :
        #   [(batch, nb * 2dim, y, x), (batch, nb * 2dim, y/2, x/2)]
        mlvl_anchors, mlvl_valid, num_level_anchors = self.anchor_generator(
            feature_shapes, batch_size=batch_size, device=device)
        self.try_to_info('anchor')
        # easy to split in batch dim or level dim.

        # cls_out_mat : tensor, (batch, Num, cls)
        # anchor_mat : tensor, (batch, Num, 2dim)
        #   - Num = total num of anchors on multi level = nb * (y * x + y/2 * x/2)
        cls_out_mat, reg_out_mat, anchor_mat, valid_mat = self.flatten_forward(
            mlvl_cls_out,
            mlvl_reg_out,
            mlvl_anchors,
            mlvl_valid)
        self.try_to_info('flat')
        # label_targets_mat : tensor, (batch, Num)
        # bbox_targets_mat  : tensor, (batch, Num, 2dim)
        label_targets_mat, bbox_targets_mat = self.get_target(
            batch_gt_labels,
            batch_gt_bboxes,
            anchor_mat,
            valid_mat
        )
        self.try_to_info('target')
        # label_targets_mat_sampled : tensor, (batch, Num)
        # bbox_targets_mat_sampled  : tensor, (batch, Num, 2dim)
        label_targets_mat_sampled, bbox_targets_mat_sampled = self.get_samples(
            cls_out_mat,
            reg_out_mat,
            label_targets_mat,
            bbox_targets_mat,
            anchor_mat,
            valid_mat,
            num_level_anchors)
        self.try_to_info('sample')
        losses = self.get_samples_losses(
            cls_out_mat,
            reg_out_mat,
            label_targets_mat_sampled,
            bbox_targets_mat_sampled,
            anchor_mat,
            valid_mat,
            num_level_anchors
        )
        losses['num_gts'] = (batch_gt_labels > 0).sum().float()
        self.try_to_info('loss')
        if self.has_proposal():
            batch_bboxes, batch_anchor_id = self.get_proposals(cls_out_mat, reg_out_mat, anchor_mat, valid_mat)
        else:
            batch_bboxes, batch_anchor_id = None, None

        if self.monitor_enable:
            # mlvl_cls_targets  : [(batch, nb, y, x), (batch, nb, y/2, x/2)]
            # mlvl_bbox_targets : [(batch, nb * 2dim, y, x), (batch, nb * 2dim, y/2, x/2)]
            mlvl_label_targets, mlvl_bbox_targets = self.flatten_backward(
                label_targets_mat,
                bbox_targets_mat,
                num_level_anchors=num_level_anchors,
                feature_shapes=feature_shapes
            )

            mlvl_label_targets_sampled, mlvl_bbox_targets_sampled = self.flatten_backward(
                label_targets_mat_sampled,
                bbox_targets_mat_sampled,
                num_level_anchors=num_level_anchors,
                feature_shapes=feature_shapes
            )

            mlvl_cls_out = list(
                map(lambda a: torch.cat([a[0], 10 * a[1], 10 * a[2]], dim=1),
                    zip(mlvl_cls_out, mlvl_label_targets, mlvl_label_targets_sampled)))
            mlvl_reg_out = list(
                map(lambda a: torch.cat(a, dim=1),
                    zip(mlvl_reg_out, mlvl_bbox_targets, mlvl_bbox_targets_sampled)))

            self.monitor_fun(mlvl_cls_out, mlvl_reg_out)
            self.try_to_info('flat back')

        return losses, batch_bboxes, (mlvl_cls_out, mlvl_reg_out)

    def forward_valid(self,
                      feats: List[torch.Tensor],
                      batch_gt_labels: torch.Tensor,
                      batch_gt_bboxes: torch.Tensor):

        mlvl_cls_out, mlvl_reg_out = self.forward(feats)

        feature_shapes = [f.size()[2:] for f in mlvl_cls_out]
        batch_size = mlvl_cls_out[0].shape[0]
        device = mlvl_cls_out[0].device

        mlvl_anchors, mlvl_valid, num_level_anchors = self.anchor_generator(
            feature_shapes, batch_size=batch_size, device=device)

        cls_out_mat, reg_out_mat, anchor_mat, valid_mat = self.flatten_forward(
            mlvl_cls_out,
            mlvl_reg_out,
            mlvl_anchors,
            mlvl_valid)

        label_targets_mat, bbox_targets_mat = self.get_target(
            batch_gt_labels,
            batch_gt_bboxes,
            anchor_mat,
            valid_mat
        )

        label_targets_mat_sampled, bbox_targets_mat_sampled = self.get_samples(
            cls_out_mat,
            reg_out_mat,
            label_targets_mat,
            bbox_targets_mat,
            anchor_mat,
            valid_mat,
            num_level_anchors)

        losses = self.get_samples_losses(
            cls_out_mat,
            reg_out_mat,
            label_targets_mat_sampled,
            bbox_targets_mat_sampled,
            anchor_mat,
            valid_mat,
            num_level_anchors
        )
        losses['num_gts'] = (batch_gt_labels > 0).sum().float()

        if self.has_proposal():
            batch_bboxes, batch_anchor_id = self.get_proposals(cls_out_mat, reg_out_mat, anchor_mat, valid_mat)
        else:
            batch_bboxes, batch_anchor_id = self.get_bboxes(cls_out_mat, reg_out_mat, anchor_mat, valid_mat)

        if self.monitor_enable:
            mlvl_label_targets, mlvl_bbox_targets = self.flatten_backward(
                label_targets_mat,
                bbox_targets_mat,
                num_level_anchors=num_level_anchors,
                feature_shapes=feature_shapes
            )

            mlvl_label_targets_sampled, mlvl_bbox_targets_sampled = self.flatten_backward(
                label_targets_mat_sampled,
                bbox_targets_mat_sampled,
                num_level_anchors=num_level_anchors,
                feature_shapes=feature_shapes
            )

            mlvl_cls_out = list(
                map(lambda a: torch.cat([a[0], 10 * a[1], 10 * a[2]], dim=1),
                    zip(mlvl_cls_out, mlvl_label_targets, mlvl_label_targets_sampled)))
            mlvl_reg_out = list(
                map(lambda a: torch.cat(a, dim=1),
                    zip(mlvl_reg_out, mlvl_bbox_targets, mlvl_bbox_targets_sampled)))

            self.monitor_fun(mlvl_cls_out, mlvl_reg_out, batch_bboxes)

        return losses, batch_bboxes, (mlvl_cls_out, mlvl_reg_out)

    def forward_infer(self, feats: List[torch.Tensor]):
        mlvl_cls_out, mlvl_reg_out = self.forward(feats)

        feature_shapes = [f.size()[2:] for f in mlvl_cls_out]
        batch_size = mlvl_cls_out[0].shape[0]
        device = mlvl_cls_out[0].device

        mlvl_anchors, mlvl_valid, num_level_anchors = self.anchor_generator(
            feature_shapes, batch_size=batch_size, device=device)

        cls_out_mat, reg_out_mat, anchor_mat, valid_mat = self.flatten_forward(
            mlvl_cls_out,
            mlvl_reg_out,
            mlvl_anchors,
            mlvl_valid)

        if self.has_proposal():
            batch_proposals, batch_anchor_id = self.get_proposals(cls_out_mat, reg_out_mat, anchor_mat, valid_mat)
            return batch_proposals, batch_anchor_id, (mlvl_cls_out, mlvl_reg_out)
        else:
            batch_bboxes, batch_anchor_id = self.get_bboxes(cls_out_mat, reg_out_mat, anchor_mat, valid_mat)
            return batch_bboxes, batch_anchor_id, (mlvl_cls_out, mlvl_reg_out)

    def monitor_fun(self, mlvl_cls_out, mlvl_reg_out, batch_bboxes=None):
        import matplotlib
        import matplotlib.pyplot as plt
        import os

        from medtk.data.visulaize import getBBox2D

        matplotlib.use('agg')

        cur_epoch = self.runner_data['epoch']
        filename = self.runner_data['img_meta'][0]['filename']
        cur_iter = self.runner_data['iter']
        result_dir = self.runner_data_meta['result_dir']
        os.makedirs(result_dir, exist_ok=True)

        data = self.runner_data

        # saving feature
        img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
        z_idx = None
        if img.ndim == 3:
            bboxes = data['gt_det'][0, :, :6].detach().cpu().numpy()
            z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2

        num_levels = len(mlvl_cls_out)
        num_scales = mlvl_cls_out[0].shape[1]  # different anchor scales

        fig, ax = plt.subplots(num_levels, num_scales + 1, figsize=(
            5 * num_scales // 1 + 5, 5 * num_levels))
        ax = ax.flatten()
        for i in range(num_levels):
            cls_score, reg_score = mlvl_cls_out[i][0].float(), mlvl_reg_out[i][0].float()
            stride = img.shape[0] // cls_score[0].shape[0]

            if img.ndim == 3:
                show = img[z_idx, ...]
            else:
                show = img

            ax0 = ax[i * (num_scales + 1)].imshow(show.astype(np.float32),
                                                  vmin=0, vmax=1)
            fig.colorbar(ax0, ax=ax[i * (num_scales + 1)])

            for j in range(num_scales):
                x = cls_score[j]
                if img.ndim == 3:
                    x = x[z_idx // stride, ...]
                x = x.sigmoid()
                x = x.detach().cpu().numpy()

                ax1 = ax[i * (num_scales + 1) + j + 1].imshow(
                    x.astype(np.float32), vmin=0, vmax=1)
                fig.colorbar(ax1, ax=ax[i * (num_scales + 1) + j + 1])

        fig = plt.gcf()
        fig.savefig(
            os.path.join(
                result_dir,
                f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewFeatures.jpg"),
            dpi=200)
        plt.close(fig)

        # saving data
        img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
        dim = img.ndim
        bboxes = data['gt_det'][0, :, :2 * dim].detach().cpu().numpy()
        labels = data['gt_det'][0, :, 2 * dim].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(5, 5))
        if dim == 2:
            show = getBBox2D(img, bboxes, labels)
            ax.imshow(show, vmin=0, vmax=1)
        else:
            z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2
            tmp_bboxes = []
            tmp_labels = []
            for idx, bbox in enumerate(bboxes):
                if bbox[2] <= z_idx <= bbox[5]:
                    tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                    tmp_labels.append(labels[idx])
            show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels)
            ax.imshow(show, vmin=0, vmax=1)
        fig = plt.gcf()
        fig.savefig(
            os.path.join(
                result_dir,
                f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewData.jpg"),
            dpi=200)
        plt.close(fig)

        if self.runner_data_meta['mode'] == 'valid':
            if not self.has_proposal():
                img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
                dim = img.ndim
                bboxes = data['gt_det'][0, :, :2 * dim].detach().cpu().numpy()
                labels = data['gt_det'][0, :, 2 * dim].detach().cpu().numpy()
                dets = batch_bboxes[0]
                dets = dets[dets[..., -1] != -1].detach().cpu().numpy()

                fig, ax = plt.subplots(1, 2, figsize=(5, 5))
                if dim == 2:
                    show = getBBox2D(img, bboxes, labels)
                    ax[0].imshow(show, vmin=0, vmax=1)
                    det_bboxes = dets[:, :4]
                    det_labels = dets[:, 4]
                    det_scores = dets[:, 5]
                    show = getBBox2D(img, det_bboxes, det_labels, det_scores)
                    ax[1].imshow(show, vmin=0, vmax=1)
                else:
                    z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2

                    tmp_bboxes = []
                    tmp_labels = []
                    for idx, bbox in enumerate(bboxes):
                        if bbox[2] <= z_idx <= bbox[5]:
                            tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                            tmp_labels.append(labels[idx])
                    show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels)
                    ax[0].imshow(show, vmin=0, vmax=1)

                    tmp_bboxes = []
                    tmp_labels = []
                    tmp_scores = []
                    for idx, bbox in enumerate(dets):
                        if bbox[2] <= z_idx <= bbox[5]:
                            tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                            tmp_labels.append(bbox[6])
                            tmp_scores.append(bbox[7])
                    show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels,
                                     tmp_scores)
                    ax[1].imshow(show, vmin=0, vmax=1)

                fig = plt.gcf()
                fig.savefig(
                    os.path.join(
                        result_dir,
                        f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewDataResults.jpg"),
                    dpi=200)
                plt.close(fig)

    def metric(self, results, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(results, ground_truth)
            metrics.update(one_metric)
        return metrics


# if __name__ == "__main__":
#     from medtk.model import *
#     from medtk.model.heads_det.utils.util_anchors import AnchorGenerator
#
#     SEED = 666
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     # anchor_generator = AnchorGenerator(
#     #     dim=2,
#     #     base_scales=1,
#     #     scales=[1, 2],
#     #     ratios=[1],
#     #     strides=[4]
#     # )
#     #
#     # feature = torch.zeros((1, 1, 24, 24))
#     # feature_shape = feature.shape[2:]
#     # mlvl_anchors, mlvl_valid, num_level_anchors = anchor_generator([feature_shape])
#     # print(mlvl_anchors[0][:, 0, 0], mlvl_valid[0][:, 0, 0])
#     # print(mlvl_anchors[0][:, 4, 4], mlvl_valid[0][:, 4, 4])
#
#     DIM = 2
#     head = AnchorHead(
#         dim=DIM,
#         in_channels=128,
#         feat_channels=128,
#         num_classes=1,
#         level_first=True,
#         anchor_generator=AnchorGenerator(
#             dim=DIM,
#             base_scales=1,
#             scales=[0.5, 1, 2, 4],
#             ratios=[1],
#             strides=[2, 4],
#             invalid_margin=4,
#         ),
#         bbox_coder=DeltaBBoxCoder(
#             dim=DIM,
#             target_stds=[1., 1., 1., 1.]),
#         assigner=UniformAssigner(
#             neg_ignore_thr=0.8 ** DIM,
#             pos_ignore_thr=0.4 ** DIM,
#             min_pos_iou=0.3 ** DIM,
#             match_low_quality=True,
#             num_neg=5000),
#         sampler=HNEMSampler(
#             pos_fraction=0.25),
#         proposal=dict(
#             nms_across_levels=False,
#             nms_pre=2000,
#             nms_post=1000,
#             max_num=1000,
#             nms_thr=0.8 ** DIM,
#             min_bbox_size=0),
#         losses=dict(
#             cls=CrossEntropyLoss(use_sigmoid=True),
#             reg=SmoothL1Loss(beta=1.0, reduction='mean', loss_weight=1.0)),
#         metrics=[
#             IOU(aggregate='none'),
#             Dist(aggregate='none', dist_threshold=5)
#         ],
#     )
#
#     print(head)
#     gt_labels = torch.tensor([
#         [1, 1, 1],
#         [1, 1, 1],
#         [1, 1, -1]
#     ]).float()
#     gt_bboxes = torch.tensor([
#         [
#             [12, 12, 18, 18],
#             [1, 1, 2, 2],
#             [1, 1, 4, 4],
#         ],
#         [
#             [12, 18, 18, 22],
#             [11, 11, 20, 20],
#             [1, 1, 4, 4],
#         ],
#         [
#             [8, 8, 16, 16],
#             [4, 4, 8, 8],
#             [-1, -1, -1, -1]
#         ]
#     ]).float()
#     features = [torch.zeros((3, 128, 24, 24)), torch.zeros((3, 128, 12, 12))]
#     loss_dict, batch_bboxes1, (mlvl_label_targets, mlvl_bbox_targets) = head.forward_train(
#         features, gt_labels, gt_bboxes)
#     print(loss_dict)
#     print(batch_bboxes1[0].shape)
#     print(batch_bboxes1[0].sum())
#
#     # import matplotlib.pyplot as plt
#     # plt.imshow(mlvl_label_targets[0][0, 2].cpu().numpy())
#     # plt.show()
#
#     # plt.imshow(mlvl_label_targets[0][1, 2].cpu().numpy())
#     # plt.show()
#     #
#     # loss_dict, batch_bboxes2, _, _ = head.forward_valid(cls_out, reg_out,
#     #                                              gt_labels, gt_bboxes)
#     # print(loss_dict)
#     # print(batch_bboxes2[0].shape)
#     #
#     # batch_bboxes = head.forward_infer(cls_out, reg_out)
#     # print(batch_bboxes[0].shape)
#     # #
