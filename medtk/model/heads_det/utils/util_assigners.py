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

import torch

from medtk.ops import iouNd_pytorch, distNd_pytorch


class IoUAssigner(torch.nn.Module):
    """
    pos_iou_thr

    neg_iou_thr

    min_pos_iou

    match_low_quality

    num_neg
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: float,
                 min_pos_iou: float = .0,
                 match_low_quality: bool = False,
                 num_neg: int = None):
        super(IoUAssigner, self).__init__()
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pos_iou_thr={self.pos_iou_thr}, neg_iou_thr={self.neg_iou_thr}, ' \
                    f'min_pos_iou={self.min_pos_iou}, match_low_quality={self.match_low_quality}, ' \
                    f'num_neg={self.num_neg})'
        return repr_str

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            iou = iouNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_iou_max, anchors_iou_argmax = torch.max(iou, dim=1)  # [num_bboxes]
                gt_iou_max, gt_iou_argmax = torch.max(iou, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(iou)
                raise

            pos_indices = anchors_iou_max >= self.pos_iou_thr
            neg_indices = anchors_iou_max < self.neg_iou_thr

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_iou_argmax).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: above positive IoU threshold
            assigned_gt_indices[pos_indices] = anchors_iou_argmax[pos_indices] + 1
            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_iou_max[i] >= self.min_pos_iou:
                        # maybe multi assign
                        # max_iou_indices = iou[:, i] == gt_iou_max[i]
                        # assigned_gt_indices[max_iou_indices] = i + 1
                        assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_iou_argmax).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class MaxIoUAssigner(torch.nn.Module):
    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: float,
                 min_pos_iou: float = .0,
                 match_low_quality: bool = False,
                 num_neg: int = None):
        super(MaxIoUAssigner, self).__init__()
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pos_iou_thr={self.pos_iou_thr}, neg_iou_thr={self.neg_iou_thr}, ' \
                    f'min_pos_iou={self.min_pos_iou}, match_low_quality={self.match_low_quality}, ' \
                    f'num_neg={self.num_neg})'
        return repr_str

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            iou = iouNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_iou_max, anchors_iou_argmax = torch.max(iou, dim=1)  # [num_bboxes]
                gt_iou_max, gt_iou_argmax = torch.max(iou, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(iou)
                raise

            pos_indices = anchors_iou_max >= self.pos_iou_thr
            neg_indices = anchors_iou_max < self.neg_iou_thr

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_iou_argmax).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: only max iou is assigned
            assigned_gt_indices[pos_indices] = -1
            for i in range(num_gts):
                if gt_iou_max[i] >= self.pos_iou_thr:
                    assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_iou_max[i] >= self.min_pos_iou:
                        # maybe multi assign
                        # max_iou_indices = iou[:, i] == gt_iou_max[i]
                        # assigned_gt_indices[max_iou_indices] = i + 1
                        assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_iou_argmax).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class DistAssigner(torch.nn.Module):
    def __init__(self,
                 pos_dist_thr: float,
                 max_pos_dist: float = 0.8,
                 match_low_quality: bool = True,
                 num_neg: int = None):
        super(DistAssigner, self).__init__()
        assert pos_dist_thr <= 1 and max_pos_dist <= 1
        self.pos_dist_thr = pos_dist_thr
        self.max_pos_dist = max_pos_dist
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pos_dist_thr={self.pos_dist_thr}, max_pos_dist={self.max_pos_dist}, ' \
                    f'match_low_quality={self.match_low_quality}, num_neg={self.num_neg})'
        return repr_str

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            dim = gt_bboxes.shape[-1] // 2
            assert dim in (2, 3)

            anchors = bboxes[..., :2 * dim]
            targets = gt_bboxes[..., :2 * dim]

            # expand dim
            anchors = torch.unsqueeze(anchors, dim=1)  # [N, 1, 2*dim]
            targets = torch.unsqueeze(targets, dim=0)  # [1, M, 2*dim]

            targets_shape = targets[..., dim:] - targets[..., :dim]  # [1, M, dim]

            anchors = (anchors[..., :dim] + anchors[..., dim:]) / 2  # [N, 1, dim]
            targets = (targets[..., :dim] + targets[..., dim:]) / 2  # [1, M, dim]

            dist = anchors - targets

            relative_dist = torch.max(torch.abs(dist / (targets_shape / 2)), dim=-1)[0]  # [N, M]

            metric = relative_dist
            # dist = distNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_dist_min, anchors_dist_argmin = torch.min(metric, dim=1)  # [num_bboxes]
                gt_dist_min, gt_dist_argmin = torch.min(metric, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(dist)
                raise

            pos_indices = anchors_dist_min <= self.pos_dist_thr
            neg_indices = anchors_dist_min >= 1

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_dist_argmin).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: above positive IoU threshold
            assigned_gt_indices[pos_indices] = anchors_dist_argmin[pos_indices] + 1
            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_dist_min[i] <= self.max_pos_dist:
                        # maybe multi assign
                        # min_dist_indices = dist[:, i] == gt_dist_min[i]
                        # assigned_gt_indices[min_dist_indices] = i + 1
                        assigned_gt_indices[gt_dist_argmin[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_dist_argmin).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class UniformAssigner(torch.nn.Module):
    """Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors, and gt_bboxes_ignore was not considered for
    now.
    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times(int): Number of positive anchors for each gt box.
           Default 4.
    """

    def __init__(self,
                 pos_ignore_thr,
                 neg_ignore_thr,
                 match_times=4,
                 min_pos_iou: float = .0,
                 match_low_quality: bool = False,
                 num_neg: int = None):
        super(UniformAssigner, self).__init__()
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pos_ignore_thr={self.pos_ignore_thr}, neg_ignore_thr={self.neg_ignore_thr}, ' \
                    f'match_times={self.match_times})'
        return repr_str

    @staticmethod
    def box_xyxy_to_cxcywh(x):
        if x.shape[-1] // 2 == 2:
            x0, y0, x1, y1 = x.unbind(-1)
            b = [(x0 + x1) / 2, (y0 + y1) / 2,
                 (x1 - x0), (y1 - y0)]
        else:
            x0, y0, z0, x1, y1, z1 = x.unbind(-1)
            b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
                 (x1 - x0), (y1 - y0), (z1 - z0)]
        return torch.stack(b, dim=-1)

    def assign(self,
               grid_bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        Args:
            grid_bboxes: with shape [N, 2dim], anchors
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]
        """
        assert grid_bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = grid_bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(grid_bboxes.device)
            assigned_bboxes = torch.zeros_like(grid_bboxes).to(grid_bboxes.device)
        else:
            # cost_bbox_pred = torch.cdist(
            #     self.box_xyxy_to_cxcywh(pred_bboxes),
            #     self.box_xyxy_to_cxcywh(gt_bboxes), p=1).cpu()  # num_bboxes, num_gts
            # cost_bbox_grid = torch.cdist(
            #     self.box_xyxy_to_cxcywh(grid_bboxes),
            #     self.box_xyxy_to_cxcywh(gt_bboxes), p=1).cpu()  # num_bboxes, num_gts

            # pred_overlaps = iouNd_pytorch(pred_bboxes, gt_bboxes)
            grid_overlaps = iouNd_pytorch(grid_bboxes, gt_bboxes)  # num_bboxes, num_gts

            # pred_max_overlaps, pred_iou_argmax = pred_overlaps.max(dim=1)
            grid_max_overlaps, grid_iou_argmax = grid_overlaps.max(dim=1)  # num_bboxes

            gt_iou_max, gt_iou_argmax = torch.max(grid_overlaps, dim=0)  # num_gts

            # self.match_times, num_gts
            # index_pred = torch.topk(- pred_overlaps, k=self.match_times, dim=0, largest=False)[1]
            index_grid = torch.topk(- grid_overlaps, k=self.match_times, dim=0, largest=False)[1]  # k, num_gts

            assigned_gt_indices = torch.ones_like(grid_iou_argmax).long() * -1  # N, 1

            # assign positive: bboxes in topk and iou > pos_ignore_thr
            for m in range(num_gts):
                index_grid[:, m][grid_overlaps[index_grid[:, m], m] < self.pos_ignore_thr] = -1
                assigned_gt_indices[index_grid[:, m][index_grid[:, m] != -1]] = m + 1

            # assign negative: bboxes not in topk and iou < neg_ignore_thr
            assigned_gt_indices[assigned_gt_indices == -1] = 1 * (
                    grid_max_overlaps[assigned_gt_indices == -1] < self.neg_ignore_thr) - 1

            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_iou_max[i] >= self.min_pos_iou:
                        assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = assigned_gt_indices.new_full((num_bboxes,), -1)
            assigned_bboxes = torch.ones_like(grid_bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class Assigner(torch.nn.Module):
    def __init__(self,
                 metrics: list = ['iou'],  # combines of iou, iof, giou, diou, abs_dist, rel_dist
                 method: str = 'all',  # one of all, max, uniform_K, atss
                 pos_criteria: str = "lambda x: 0.5 <= x['iou']",
                 neg_criteria: str = "lambda x: x['iou'] < 0.3",
                 min_pos_criteria: str = "lambda x: 0.2 <= x['iou']",
                 match_low_quality: bool = False,
                 num_neg: int = None):
        super(Assigner, self).__init__()
        assert set(metrics).issubset({'iou', 'iof', 'giou', 'diou', 'abs_dist', 'rel_dist'})
        assert method in ['all', 'max'] or method.startswith('uniform_')

        self.metrics = metrics
        self.method = method
        self.pos_criteria = eval(pos_criteria)
        self.neg_criteria = eval(neg_criteria)
        self.min_pos_criteria = eval(min_pos_criteria)
        self._pos_criteria = pos_criteria
        self._neg_criteria = neg_criteria
        self._min_pos_criteria = min_pos_criteria
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(metrics={self.metrics}, method={self.method}, ' \
                    f'pos_criteria="{self._pos_criteria}", ' \
                    f'neg_criteria="{self._neg_criteria}", ' \
                    f'min_pos_criteria="{self._min_pos_criteria}", ' \
                    f'match_low_quality={self.match_low_quality}, num_neg={self.num_neg})'
        return repr_str

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
            return assigned_labels, assigned_bboxes

        metrics = dict(
            anchors_metric_max={},
            anchors_metric_argmax={},
            gt_metric_max={},
            gt_metric_argmax={},
        )
        for metric in self.metrics:
            if metric in ['iou', 'iof', 'giou', 'diou']:
                # 0. calculate iou
                metric_anchor_gt = iouNd_pytorch(bboxes, gt_bboxes, mode=metric)  # [num_bboxes, num_gts]
            elif metric in ['abs_dist', 'rel_dist']:
                # 0. calculate dist
                metric_anchor_gt = distNd_pytorch(bboxes, gt_bboxes, mode=metric)  # [num_bboxes, num_gts]

            else:
                raise NotImplementedError

            metrics[metric] = metric_anchor_gt

        pos_indices, pos_indices_argmax = torch.max(self.pos_criteria(metrics), dim=1)
        neg_indices, neg_indices_argmax = torch.min(self.neg_criteria(metrics), dim=1)

        if self.method in ['all', 'max']:
            # 1. assign -1 to each bboxes
            assigned_gt_indices = bboxes.new_ones(bboxes.shape[0]).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            if self.method == 'max':
                # 3. assign positive: only max iou is assigned
                assigned_gt_indices[pos_indices] = -1
            elif self.method == 'all':
                # 3. assign positive: above positive IoU threshold
                assigned_gt_indices[pos_indices] = pos_indices_argmax[pos_indices] + 1
        elif self.method.startswith('uniform_'):
            match_times = int(self.method.replace('uniform_', ''))
            if 'iou' in self.metrics:
                grid_overlaps = metrics['iou']
            else:
                grid_overlaps = iouNd_pytorch(bboxes, gt_bboxes)  # num_bboxes, num_gts
            grid_max_overlaps, grid_iou_argmax = grid_overlaps.max(dim=1)  # num_bboxes
            index_grid = torch.topk(- grid_overlaps, k=match_times, dim=0, largest=False)[1]  # [K, M]

            # 1. assign -1 to each bboxes
            assigned_gt_indices = bboxes.new_ones(bboxes.shape[0]).long() * -1
            # 2. assign positive: bboxes in topk and iou match pos_criteria
            for m in range(num_gts):
                assigned_gt_indices[index_grid[:, m][self.pos_criteria(metrics)[index_grid[:, m], m]]] = m + 1

            # 3. assign negative: bboxes not in topk and iou match neg_criteria
            assigned_gt_indices[assigned_gt_indices == -1] = 1 * (
                neg_indices[assigned_gt_indices == -1]) - 1
        else:
            raise NotImplementedError

        # 4. assign low quality gt to best anchors
        if self.match_low_quality:
            for i in range(num_gts):
                gt_min_pos_indices, gt_min_pos_indices_argmax = torch.max(self.min_pos_criteria(metrics), dim=0)
                if gt_min_pos_indices[i]:
                    # maybe multi assign
                    # max_iou_indices = iou[:, i] == gt_iou_max[i]
                    # assigned_gt_indices[max_iou_indices] = i + 1
                    assigned_gt_indices[gt_min_pos_indices_argmax[i]] = i + 1

        pos_indices = torch.where(assigned_gt_indices > 0)[0]
        neg_indices = torch.where(assigned_gt_indices == 0)[0]

        if self.num_neg is not None:
            neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

        assigned_labels = bboxes.new_ones(bboxes.shape[0]).long() * -1
        assigned_bboxes = torch.ones_like(bboxes) * -1.0
        if pos_indices.numel() > 0:
            assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
            assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
        assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(suppress=True)

    anchors = torch.tensor(np.array([[2.0, 2.0, 4.0, 4.0],
                                     [2.0, 2.0, 5.0, 4.0],
                                     [1.0, 1.0, 3.0, 4.0],
                                     [1.0, 1.0, 3.5, 2.5],
                                     [1.0, 1.0, 3.0, 2.5],
                                     [8.0, 8.0, 18.0, 18.0],
                                     [8.1, 8.2, 18.0, 18.0],
                                     [8.0, 8.2, 18.1, 18.2],
                                     [8.1, 8.2, 18.0, 18.2],
                                     [8.0, 8.2, 18.1, 18.2],
                                     [9.0, 9.2, 16.1, 16.2],
                                     [9.0, 9.2, 14.1, 14.2],
                                     [9.0, 9.2, 12.1, 12.2],
                                     [20.0, 20.0, 40.0, 40.0],
                                     [20.0, 21.1, 41.0, 40.0],
                                     [21.0, 20.0, 40.1, 41.0],
                                     [20.0, 21.2, 41.0, 40.0],
                                     [21.0, 20.0, 40.2, 41.0],
                                     [20.0, 21.3, 41.0, 40.0],
                                     [21.0, 20.0, 40.3, 41.0],
                                     [21.0, 20.0, 30.3, 31.0],
                                     [21.0, 20.0, 24.3, 25.0],
                                     ]))
    gt_bboxes = torch.tensor(np.array([[2.0, 2.0, 4.0, 4.0],
                                       [8.0, 8.0, 18.0, 18.0],
                                       [20.0, 20.0, 40.0, 40.0]]))
    gt_labels = torch.tensor(np.array([1, 2, 3], dtype=np.int)).long()

    iou = iouNd_pytorch(anchors, gt_bboxes, mode='iou')
    print(iou.cpu().numpy())
    iof = iouNd_pytorch(anchors, gt_bboxes, mode='iof')
    print(iof.cpu().numpy())
    dist = distNd_pytorch(anchors, gt_bboxes, mode='abs_dist')
    print(dist.cpu().numpy())
    dist = distNd_pytorch(anchors, gt_bboxes, mode='rel_dist')
    print(dist.cpu().numpy())

    # assigner = UniformAssigner(pos_ignore_thr=0.3, neg_ignore_thr=0.7, match_times=4, min_pos_iou=0.2,
    #                            match_low_quality=True)
    # assigned_labels, assigned_bboxes = assigner.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # print(assigner)
    # print(assigned_labels)
    # print(torch.cat([anchors, assigned_bboxes], dim=1))

    # assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=0.2, match_low_quality=True)
    # assigned_labels, assigned_bboxes = assigner.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # print(assigner)
    # print(assigned_labels)
    # print(torch.cat([anchors, assigned_bboxes], dim=1))

    # assigner = IoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=0.2, match_low_quality=True)
    # assigned_labels, assigned_bboxes = assigner.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # print(assigner)
    # print(assigned_labels)
    # print(torch.cat([anchors, assigned_bboxes], dim=1))

    # assigner = Assigner(metrics=['iou', 'iof', 'rel_dist'],
    #                     method='uniform_4',
    #                     pos_criteria="lambda x: (0.5 <= x['iou']) * (0.2 < x['iof']) * (x['rel_dist'] < 0.9)",
    #                     neg_criteria="lambda x: x['iou'] < 0.3",
    #                     min_pos_criteria="lambda x: 0.2 <= x['iou']",
    #                     match_low_quality=True)
    # assigned_labels, assigned_bboxes = assigner.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # print(assigner)
    # print(assigned_labels)
    # print(torch.cat([anchors, assigned_bboxes], dim=1))

    points = torch.tensor(np.array([[3.2, 3.0, 3.2, 3.0],
                                    [3.5, 3.0, 3.5, 3.0],
                                    [2.0, 2.5, 2.0, 2.5],
                                    [2.2, 1.7, 2.2, 1.7],
                                    [1.0, 1.7, 1.0, 1.7],
                                    [13.0, 13.0, 13.0, 13.0],
                                    [13.0, 13.1, 13.0, 13.1],
                                    [13.0, 13.2, 13.0, 13.2],
                                    [13.0, 13.2, 13.0, 13.2],
                                    [13.0, 13.2, 13.0, 13.2],
                                    [12.5, 12.7, 12.5, 12.7],
                                    [11.5, 11.7, 11.5, 11.7],
                                    [10.5, 10.7, 10.5, 10.7],
                                    [30.0, 30.0, 30.0, 30.0],
                                    [30.5, 30.5, 30.5, 30.5],
                                    [30.5, 30.5, 30.5, 30.5],
                                    [30.5, 30.6, 30.5, 30.6],
                                    [30.6, 30.5, 30.6, 30.5],
                                    [30.5, 30.6, 30.5, 30.6],
                                    [30.6, 30.5, 30.6, 30.5],
                                    [25.6, 25.5, 25.6, 25.5],
                                    [22.6, 22.5, 22.6, 22.5],
                                    ]))
    gt_bboxes = torch.tensor(np.array([[2.0, 2.0, 4.0, 4.0],
                                       [8.0, 8.0, 18.0, 18.0],
                                       [20.0, 20.0, 40.0, 40.0]]))
    assigner = DistAssigner(pos_dist_thr=0.5, max_pos_dist=1.0, match_low_quality=True)
    assigned_labels, assigned_bboxes = assigner.assign(points, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    print(assigner)
    print(assigned_labels)
    print(torch.cat([points, assigned_bboxes], dim=1))
