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

from medtk.model.nnModules import ComponentModule

from medtk.model.heads_det.utils.util_bboxes import iouNd_pytorch


class IOU(ComponentModule):
    def __init__(self,
                 task: str = 'det',
                 iou_per_dim=0.7,
                 aggregate=None):
        super().__init__()
        assert aggregate in [None, 'mean', 'sum', 'none']
        assert task.upper() in ('DET', 'SEG')
        self.task = task.upper()
        self.aggregate = aggregate is None and 'none' or aggregate
        self.iou_per_dim = iou_per_dim

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(task={self.task}, aggregate={self.aggregate})'
        return repr_str

    def __call__(self, predict, target):
        if self.task == 'SEG':
            return self._iou_seg(predict, target)
        else:
            return self._iou_det(predict, target)

    def _iou_det(self,
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
        iou_threshold = self.iou_per_dim ** dim
        batch_size = len(batch_gt_bboxes)
        recall, precision, miou, mscore, dets, gts, matched = torch.zeros(7).to(batch_gt_bboxes.device)
        for i, (bboxes, gt_bboxes) in enumerate(zip(batch_bboxes, batch_gt_bboxes)):
            bboxes = bboxes[bboxes[..., -1] != -1].float().cpu()
            gt_bboxes = gt_bboxes[gt_bboxes[..., -1] != -1].float().cpu()

            self.try_to_info("bboxes", bboxes)
            self.try_to_info("gt_bboxes", gt_bboxes)
            num_bboxes, num_gts = len(bboxes), len(gt_bboxes)
            if not num_bboxes:
                recall += 0
                precision += 0
                miou += 0
                mscore += 0
                dets += num_bboxes
                gts += num_gts
                matched += 0
                continue

            IOU = iouNd_pytorch(bboxes, gt_bboxes)
            bboxes_iou_max, bboxes_iou_argmax = torch.max(IOU, dim=1)  # [num_bboxes]
            gt_iou_max, gt_iou_argmax = torch.max(IOU, dim=0)  # [num_gts]
            self.try_to_info("IOU bboxes", bboxes_iou_max)
            self.try_to_info("IOU gt", gt_iou_max)
            TP = (gt_iou_max > iou_threshold).sum().float()
            if TP > 0:
                mscore += bboxes[:, -1][gt_iou_argmax[gt_iou_max > iou_threshold]].mean()
                min_score = bboxes[:, -1][gt_iou_argmax[gt_iou_max > iou_threshold]].min()
            else:
                mscore += 0
                min_score = 0
            recall += TP / num_gts
            precision += TP / (bboxes[:, -1] >= min_score).sum()
            miou += gt_iou_max.mean()
            dets += num_bboxes
            gts += num_gts
            matched += TP

        metric = {'iou_recall':      recall / batch_size,
                  'iou_precision':   precision / batch_size,
                  'iou_mean':        miou / batch_size,
                  'iou_mean_score':  mscore / batch_size,
                  'iou_dets':        dets / batch_size,
                  'iou_gts':         gts / batch_size,
                  'iou_matched':     matched / batch_size}
        # print(metric)
        return metric

    def _iou_seg(self,
                 net_output: torch.Tensor,
                 target: torch.Tensor):
        """

        Args:
            net_output: b,c,d,h,w
            target:     b,1,d,h,w

        Returns:

        """

        assert net_output.ndim == target.ndim, f'Dimension not matched! {net_output.shape}, {target.shape}'

        batch_size = net_output.shape[0]
        num_classes = net_output.shape[1]

        target = target.view(batch_size, -1)
        prediction = torch.argmax(net_output, dim=1, keepdim=True).view(batch_size, -1)
        assert target.shape == prediction.shape

        IOU = torch.zeros(num_classes).to(target.device)
        for i in range(num_classes):
            prediction_i = (prediction == i).cpu().float()
            target_i = ((target == i) * 1).cpu().float()

            smooth = 1.
            intersection_i = torch.sum(prediction_i * target_i)
            union_i = prediction_i.sum() + target_i.sum() - intersection_i
            iou_i = torch.true_divide(intersection_i + smooth, union_i + smooth)
            IOU[i] = iou_i
        if self.aggregate == 'none':
            metric = dict(zip([f"iou_class_{i}" for i in range(num_classes)], IOU))
        elif self.aggregate == 'mean':
            metric = {'mean_iou': torch.mean(IOU)}
        else:
            metric = {'sum_iou': torch.sum(IOU)}
        return metric


if __name__ == '__main__':
    m = IOU()

    bboxes = torch.tensor([
        [[1, 1, 8, 8, 1, 0.9],
         [1, 1, 2, 2, 1, 0.6],
         [1, 1, 3, 2, 1, 0.3]],
    ])

    gt_bboxes = torch.tensor([
        [[1, 1, 2, 2, 1, 0.6]],
    ])
    print(m(bboxes, gt_bboxes))