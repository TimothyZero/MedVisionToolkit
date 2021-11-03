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

from medtk.model.nnModules import ComponentModule

from medtk.ops import distNd_pytorch


class Dist(ComponentModule):
    def __init__(self,
                 task: str = 'det',
                 dist_threshold=5,
                 aggregate=None):
        super().__init__()
        assert aggregate in [None, 'mean', 'sum', 'none']
        assert task.upper() in ('DET', 'SEG')
        self.task = task.upper()
        self.dist_threshold = dist_threshold
        self.aggregate = aggregate is None and 'none' or aggregate

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(task={self.task}, aggregate={self.aggregate})'
        return repr_str

    def __call__(self, predict, target):
        if self.task == 'SEG':
            return self._dist_seg(predict, target)
        else:
            return self._dist_det(predict, target)

    def _dist_det(self,
                  batch_bboxes: torch.Tensor,
                  batch_gt_bboxes: torch.Tensor):
        """

        Args:
            batch_bboxes:
            batch_gt_bboxes:

        Returns:

        """
        dim = batch_gt_bboxes.shape[-1] // 2 - 1
        batch_gt_bboxes = batch_gt_bboxes[..., :2 * dim]
        batch_size = len(batch_gt_bboxes)
        recall, precision, mdist, mscore, dets, gts, matched = torch.zeros(7).to(batch_bboxes.device)
        for i, (bboxes, gt_bboxes) in enumerate(zip(batch_bboxes, batch_gt_bboxes)):
            bboxes = bboxes[bboxes[..., -1] != -1].float().cpu()
            gt_bboxes = gt_bboxes[gt_bboxes[..., -1] != -1].float().cpu()

            self.try_to_info("bboxes", bboxes)
            self.try_to_info("gt_bboxes", gt_bboxes)
            num_bboxes, num_gts = len(bboxes), len(gt_bboxes)
            if not num_bboxes:
                recall += 0
                precision += 0
                mdist += 0
                dets += num_bboxes
                gts += num_gts
                matched += 0
                continue

            DIST = distNd_pytorch(bboxes, gt_bboxes)
            bboxes_dist_min, bboxes_dist_argmin = torch.min(DIST, dim=1)  # [num_bboxes]
            gt_dist_min, gt_dist_argmin = torch.min(DIST, dim=0)  # [num_gts]
            self.try_to_info("DIST bboxes", bboxes_dist_min)
            self.try_to_info("DIST gt", gt_dist_min)
            TP = (gt_dist_min < self.dist_threshold).sum().float()
            if TP > 0:
                mscore += bboxes[:, -1][gt_dist_argmin[gt_dist_min < self.dist_threshold]].mean()
                min_score = bboxes[:, -1][gt_dist_argmin[gt_dist_min < self.dist_threshold]].min()
            else:
                mscore += 0
                min_score = 0
            recall += TP / num_gts
            precision += TP / (bboxes[:, -1] >= min_score).sum()
            mdist += gt_dist_min.mean()
            dets += num_bboxes
            gts += num_gts
            matched += TP

        metric = {'dist_recall':      recall / batch_size,
                  'dist_precision':   precision / batch_size,
                  'dist_mean':        mdist / batch_size,
                  'dist_mean_score':  mscore / batch_size,
                  'dist_dets':        dets / batch_size,
                  'dist_gts':         gts / batch_size,
                  'dist_matched':     matched / batch_size}
        # print(metric)
        return metric


if __name__ == '__main__':
    m = Dist(dist_threshold=2)

    bboxes = torch.tensor([
        [[1, 1, 8, 8, 1, 0.9],
         [1, 1, 2, 2, 1, 0.6],
         [1, 1, 3, 2, 1, 0.3]],
    ])

    gt_bboxes = torch.tensor([
        [[1, 1, 2, 2, 1, 0.6]],
    ])
    print(m(bboxes, gt_bboxes))