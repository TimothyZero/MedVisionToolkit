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
from typing import Union, List
from torch import nn
import torch
import numpy as np
import math
import warnings

from medtk.model.nnModules import ComponentModule

from medtk.ops import nmsNd_pytorch, softnmsNd_pytorch, bbox2roi
from medtk.utils import multi_apply

try:
    from medvision.ops import nms_nd as nmsNd_cuda
except ImportError:
    warnings.warn('no nms cuda!')
    from medtk.ops import nmsNd_pytorch as nmsNd_cuda


class BBoxHead(ComponentModule):
    def __init__(self,
                 dim,
                 num_classes,
                 in_channels,
                 feat_channels: int = 256,
                 fc_channels: int = 1024,
                 roi_output_size: int = 7,
                 assigner: dict = None,
                 sampler: dict = None,
                 bbox_coder: dict = None,
                 nms: dict = None,
                 losses: dict = None,
                 metrics: Union[List[dict], List[object]] = None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(BBoxHead, self).__init__(conv_cfg, norm_cfg, act_cfg)

        if losses.get('cls', None) is None:
            losses['cls'] = dict(type='CrossEntropyLoss')
        if losses.get('reg', None) is None:
            losses['reg'] = dict(type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0)

        self.dim = dim
        self.in_channels = in_channels  # using sum instead of cat
        self.num_classes = num_classes
        self.roi_output_size = (roi_output_size,) * self.dim
        self.feature_channels = feat_channels
        self.fc_channels = fc_channels
        self.flatten_features = self.feature_channels * np.prod(np.array(self.roi_output_size))

        self.bboxCoder = bbox_coder

        self.assigner = assigner
        self.sampler = sampler
        self.nms = nms

        self.extractors = None

        self.criterion_cls = losses['cls']
        self.criterion_reg = losses['reg']
        self.metrics = nn.ModuleList([metric for metric in metrics])

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.shared_convs = nn.Sequential(
            self.build_conv(self.dim, self.in_channels, self.feature_channels, kernel_size=3, padding=1),
            self.build_act(self.act_cfg),
        )
        self.shared_fcs = nn.Linear(self.flatten_features, self.fc_channels)
        self.roi_cls = nn.Linear(self.fc_channels, self.num_classes + 1)
        self.roi_reg = nn.Linear(self.fc_channels, 2 * self.dim)

    def _init_weights(self):
        pass

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())

        roi_features = self.extractors(feats, rois)

        out = self.shared_convs(roi_features)
        out = out.view(-1, self.flatten_features)
        out = self.shared_fcs(out)
        cls_out = self.roi_cls(out)
        reg_out = self.roi_reg(out)
        return cls_out, reg_out

    def set_extractor(self, extractors):
        self.extractors = extractors

    def forward_train(self, multi_level_features, batch_proposals, batch_gt_labels, batch_gt_bboxes):
        batch_multi_level_features = list(map(lambda a: [i[None, ...] for i in a], zip(*multi_level_features)))

        batch_rois, batch_labels, batch_deltas, batch_proposal_pos = multi_apply(
            self._target_single_image,
            batch_proposals,
            batch_gt_labels,
            batch_gt_bboxes,
            batch_multi_level_features
        )
        rois = bbox2roi(batch_rois)
        rois_deltas = torch.cat(batch_deltas, dim=0)
        rois_labels = torch.cat(batch_labels, dim=0)

        proposal_pos_fraction = sum(batch_proposal_pos) / sum([len(i) for i in batch_proposals])

        cls, reg = self._bbox_forward(multi_level_features, rois)
        roi_loss_dict = self.get_losses(cls, reg, rois_labels, rois_deltas)
        roi_loss_dict['proposal_pos_fraction'] = proposal_pos_fraction
        return roi_loss_dict

    def forward_valid(self, multi_level_features, batch_proposals, batch_gt_labels, batch_gt_bboxes):
        batch_multi_level_features = list(map(lambda a: [i[None, ...] for i in a], zip(*multi_level_features)))

        batch_rois, batch_labels, batch_deltas, batch_proposal_pos = multi_apply(
            self._target_single_image,
            batch_proposals,
            batch_gt_labels,
            batch_gt_bboxes,
            batch_multi_level_features
        )
        rois = bbox2roi(batch_rois)
        rois_deltas = torch.cat(batch_deltas, dim=0)
        rois_labels = torch.cat(batch_labels, dim=0)

        proposal_pos_fraction = sum(batch_proposal_pos) / sum([len(i) for i in batch_proposals])

        cls, reg = self._bbox_forward(multi_level_features, rois)
        roi_loss_dict = self.get_losses(cls, reg, rois_labels, rois_deltas)
        roi_loss_dict['proposal_pos_fraction'] = proposal_pos_fraction
        batch_bboxes = self.get_bboxes(rois, cls, reg)

        if self.monitor_enable:
            self.monitor_fun(batch_bboxes)

        return roi_loss_dict, batch_bboxes

    def forward_infer(self, multi_level_features, batch_proposals, batch_anchor_id: list):
        rois = bbox2roi(batch_proposals)
        rois_anchor_id = torch.cat(batch_anchor_id, dim=0)

        cls, reg = self._bbox_forward(multi_level_features, rois)

        batch_bboxes, batch_roi_anchor_id = self.get_bboxes_refined(rois, rois_anchor_id, cls, reg)
        # batch_bboxes = self.get_bboxes(rois, cls, reg)
        return batch_bboxes, batch_roi_anchor_id

    def monitor_fun(self, batch_bboxes):
        import matplotlib
        import matplotlib.pyplot as plt
        import os

        from medtk.data.visulaize import getBBox2D

        matplotlib.use('agg')

        cur_epoch = self.runner_data['epoch'] + 1
        filename = self.runner_data['img_meta'][0]['filename']
        cur_iter = self.runner_data['iter']
        result_dir = self.runner_data_meta['result_dir']
        os.makedirs(result_dir, exist_ok=True)

        data = self.runner_data

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

    def _target_single_image(self, proposals, gt_labels, gt_bboxes, multi_level_features):
        gt_bboxes = gt_bboxes[gt_labels != -1]
        gt_labels = gt_labels[gt_labels != -1].long()

        self.try_to_info("=========")
        self.try_to_info(gt_bboxes.shape, gt_labels.shape)
        self.try_to_info("labels info", gt_bboxes, gt_labels)
        self.try_to_info(proposals.shape)

        assigned_labels, assigned_bboxes = self.assigner.assign(proposals, gt_bboxes, gt_labels)
        self.try_to_info(proposals[-10:])
        self.try_to_info(assigned_bboxes[-10:])
        self.try_to_info(assigned_labels[-10:])
        self.try_to_info("Positive", torch.sum(assigned_labels > 0))

        proposal_pos = torch.sum(assigned_labels > 0)

        proposed_rois = bbox2roi([proposals])
        _cls, _ = self._bbox_forward(multi_level_features, proposed_rois)

        losses = torch.zeros_like(assigned_labels).float()
        with torch.no_grad():
            not_ignored = assigned_labels >= 0
            losses[not_ignored] = self.criterion_cls(_cls[not_ignored],
                                                     assigned_labels[not_ignored], reduction_override='none').squeeze(
                -1)

        pos_indices, neg_indices, proposals, assigned_labels, assigned_bboxes = self.sampler.sample(proposals,
                                                                                                    assigned_labels,
                                                                                                    assigned_bboxes,
                                                                                                    gt_labels,
                                                                                                    gt_bboxes,
                                                                                                    weight=losses)
        self.try_to_info(assigned_labels.shape, assigned_bboxes.shape, proposals.shape)
        self.try_to_info(len(pos_indices))
        # pos_indices = torch.where(assigned_labels > 0)[0]
        assigned_bboxes[pos_indices] = self.bboxCoder.encode(proposals[pos_indices], assigned_bboxes[pos_indices])

        """"""""""""""""""""""""""
        "set target"
        indices = torch.cat([pos_indices, neg_indices], dim=0)
        proposal_sample = proposals[indices]
        proposal_label = assigned_labels[indices]
        proposal_deltas = assigned_bboxes[indices]

        self.try_to_info("shape1", proposal_sample.shape, proposal_label.shape, proposal_deltas.shape)

        # 添加rois,deltas,labels
        self.try_to_info("shape2", proposal_sample.shape, proposal_label.shape, proposal_deltas.shape)
        return proposal_sample, proposal_label, proposal_deltas, proposal_pos

    def get_losses(self, cls_out, reg_out, rois_labels, rois_deltas):
        # self.try_to_info("batch_proposals_cls", batch_proposals_cls[:10])
        # self.try_to_info("batch_proposals_cls", cls_out[:10, :])
        # self.try_to_info("batch_proposals_reg", batch_proposals_reg[:10, :])
        # self.try_to_info("batch_proposals_reg", reg_out[:10, :])
        self.try_to_info("shape", cls_out.shape, rois_labels.shape,
                         reg_out.shape, rois_deltas.shape)

        loss_cls = self.criterion_cls(cls_out, rois_labels.long())
        # self.try_to_info("1")
        # loss_cls_2 = torch.nn.CrossEntropyLoss()(cls_out, batch_proposals_cls.long())
        # self.try_to_info(loss_cls, loss_cls_2)

        roi_num_neg = (rois_labels == 0).sum().to(cls_out.device)
        roi_num_pos = (rois_labels != 0).sum().to(cls_out.device)

        pred_cls = torch.argmax(cls_out, dim=1)
        roi_acc_all = 100.0 * torch.mean((pred_cls == rois_labels).float())
        roi_acc_pos = 100.0 * torch.mean((pred_cls[rois_labels > 0] == rois_labels[rois_labels > 0]).float())

        if rois_labels.sum() > 0:
            pos_reg_out = reg_out[rois_labels > 0]
            target_reg = rois_deltas[rois_labels > 0]
            self.try_to_info(pos_reg_out.shape, target_reg.shape)
            loss_reg = self.criterion_reg(pos_reg_out, target_reg)
            # self.try_to_info(pos_reg_out.shape, target_reg.shape)
        else:
            loss_reg = torch.tensor(0.0).to(cls_out.device)

        self.try_to_info(loss_cls, loss_reg)
        return dict(
            roi_reg_loss=loss_reg,
            roi_cls_loss=loss_cls,
            roi_num_neg=roi_num_neg,
            roi_num_pos=roi_num_pos,
            roi_trained=torch.tensor(rois_labels.size(0)).to(cls_out.device),
            roi_acc_all=roi_acc_all,
            roi_acc_pos=roi_acc_pos
            )

    def get_bboxes(self, rois, cls, reg):
        batch = int(torch.max(rois[:, 0]).item() + 1)
        self.try_to_info('batch', batch)
        batch_rois, batch_cls, batch_reg = [], [], []
        for i in range(batch):
            batch_rois.append(rois[rois[:, 0] == i])
            batch_cls.append(cls[rois[:, 0] == i])
            batch_reg.append(reg[rois[:, 0] == i])

        batch_bboxes = multi_apply(
            self._bboxes_single_image,
            batch_rois,
            batch_cls,
            batch_reg
        )[0]
        batch_bboxes = torch.tensor(batch_bboxes).float().to(cls.device)
        return batch_bboxes

    def _bboxes_single_image(self, roi, cls, reg):
        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            if roi.is_cuda:
                nms_fun = nmsNd_cuda
            else:
                nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        cls_out_flat = torch.softmax(cls, dim=1)
        scores, labels = torch.max(cls_out_flat[:, 1:], dim=1)
        bboxes = self.bboxCoder.decode(roi[:, 1:], reg)
        self.try_to_info('roi', roi)
        self.try_to_info('bboxes', bboxes)

        nms_idx, _ = nms_fun(torch.cat([bboxes, scores.unsqueeze(-1)], dim=1), iou_threshold)
        scores, labels, bboxes = scores[nms_idx], labels[nms_idx], bboxes[nms_idx]

        high_score_indices = np.where(scores.cpu() > score_threshold)[0]

        results = np.ones((max_per_img, self.dim * 2 + 2)) * -1
        for j in range(min(high_score_indices.shape[0], max_per_img)):
            bbox = bboxes[high_score_indices[j]]
            bbox = bbox.detach().cpu().numpy()
            label = int(labels[high_score_indices[j]].item()) + 1
            score = scores[high_score_indices[j]].item()
            self.try_to_info("postprocess", [*bbox, label, score])
            results[j] = [*bbox, label, score]

        return [results]

    def get_bboxes_refined(self, rois, rois_anchor_id, cls, reg):
        batch = int(torch.max(rois[:, 0]).item() + 1)
        self.try_to_info('batch', batch)
        batch_rois, batch_rois_anchor_id, batch_cls, batch_reg = [], [], [], []
        for i in range(batch):
            batch_rois.append(rois[rois[:, 0] == i])
            batch_rois_anchor_id.append(rois_anchor_id[rois[:, 0] == i])
            batch_cls.append(cls[rois[:, 0] == i])
            batch_reg.append(reg[rois[:, 0] == i])

        batch_bboxes, batch_roi_anchor_id = multi_apply(
            self._bboxes_refined_single_image,
            batch_rois,
            batch_rois_anchor_id,
            batch_cls,
            batch_reg
        )
        batch_bboxes = torch.tensor(batch_bboxes).float().to(cls.device)
        batch_roi_anchor_id = torch.tensor(batch_roi_anchor_id, dtype=torch.int64).to(cls.device)
        return batch_bboxes, batch_roi_anchor_id

    def _bboxes_refined_single_image(self, roi, roi_anchor_id, cls, reg):
        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            if roi.is_cuda:
                nms_fun = nmsNd_cuda
            else:
                nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        cls_out_flat = torch.softmax(cls, dim=1)
        scores, labels = torch.max(cls_out_flat[:, 1:], dim=1)
        bboxes = self.bboxCoder.decode(roi[:, 1:], reg)
        self.try_to_info('roi', roi)
        self.try_to_info('bboxes', bboxes)

        keep, _ = nms_fun(torch.cat([bboxes, scores.unsqueeze(-1)], dim=1), iou_threshold)
        scores, labels, bboxes, roi_anchor_id = scores[keep], labels[keep], bboxes[keep], roi_anchor_id[keep]

        high_score_indices = np.where(scores.cpu() > score_threshold)[0]

        results = np.ones((max_per_img, self.dim * 2 + 2)) * -1
        roi_anchor_ids = np.ones((max_per_img, ), dtype=np.int) * -1
        for j in range(min(high_score_indices.shape[0], max_per_img)):
            bbox = bboxes[high_score_indices[j]]
            bbox = bbox.detach().cpu().numpy()
            label = int(labels[high_score_indices[j]].item()) + 1
            score = scores[high_score_indices[j]].item()
            self.try_to_info("postprocess", [*bbox, label, score])
            results[j] = [*bbox, label, score]
            roi_anchor_ids[j] = roi_anchor_id[high_score_indices[j]].item()

        return results, roi_anchor_ids

    def metric(self, results, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(results, ground_truth)
            metrics.update(one_metric)
        return metrics
