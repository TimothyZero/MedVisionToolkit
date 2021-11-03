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

import torch
import torch.nn.functional as F

from medtk.ops import iouNd_pytorch, deltaNd_pytorch, applyDeltaNd_pytorch


class DeltaBBoxCoder(torch.nn.Module):
    """
    References: mmdet
    Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 dim,
                 target_means=None,
                 target_stds=None):
        super().__init__()
        self.dim = dim
        if target_means is None:
            target_means = [.0] * 2 * dim
        if target_stds is None:
            target_stds = [.1] * dim + [.2] * dim
        if isinstance(target_means, (list, tuple)):
            target_means = torch.tensor(target_means)
        if isinstance(target_stds, (list, tuple)):
            target_stds = torch.tensor(target_stds)

        self.means = target_means
        self.stds = target_stds

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += f'means={self.means.cpu().numpy().tolist()}, '
        repr_str += f'stds={self.stds.cpu().numpy().tolist()}'
        repr_str += ')'
        return repr_str

    def encode(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] // 2 == gt_bboxes.shape[-1] // 2 == self.dim
        bboxes = bboxes[..., :2 * self.dim]
        gt_bboxes = gt_bboxes[..., :2 * self.dim]
        encoded_bboxes = deltaNd_pytorch(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, deltas):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes.
            deltas (torch.Tensor): Encoded boxes with shape

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert deltas.shape[0] == bboxes.shape[0], f"{deltas.shape}, {bboxes.shape}"
        bboxes = bboxes[..., :2 * self.dim]
        deltas = deltas[..., :2 * self.dim]
        decoded_bboxes = applyDeltaNd_pytorch(bboxes, deltas, self.means, self.stds)

        return decoded_bboxes


class DistBBoxCoder(torch.nn.Module):
    def __init__(self,
                 dim,
                 target_means=None,
                 target_stds=None):
        super().__init__()
        self.dim = dim
        if target_means is None:
            target_means = [.0] * 2 * dim
        if target_stds is None:
            target_stds = [.1] * dim + [.2] * dim
        if isinstance(target_means, (list, tuple)):
            target_means = torch.tensor(target_means)
        if isinstance(target_stds, (list, tuple)):
            target_stds = torch.tensor(target_stds)

        self.means = target_means
        self.stds = target_stds

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += f'means={self.means.cpu().numpy().tolist()}, '
        repr_str += f'stds={self.stds.cpu().numpy().tolist()}'
        repr_str += ')'
        return repr_str

    def encode(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] // 2 == gt_bboxes.shape[-1] // 2 == self.dim
        assert torch.all(bboxes[:10, :self.dim] == bboxes[:10, self.dim:2 * self.dim]), 'anchors should be points'

        anchors = bboxes[..., :2 * self.dim]
        targets = gt_bboxes[..., :2 * self.dim]

        encoded_bboxes = torch.abs(anchors - targets)  # [N, 2dim]
        return encoded_bboxes

    def decode(self, bboxes, deltas):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes.
            deltas (torch.Tensor): Encoded boxes with shape

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert deltas.shape[0] == bboxes.shape[0], f"{deltas.shape}, {bboxes.shape}"
        assert torch.all(bboxes[:10, :self.dim] == bboxes[:10, self.dim:2 * self.dim]), 'anchors should be points'

        anchors = bboxes[..., :2 * self.dim]
        deltas = F.relu(deltas[..., :2 * self.dim])
        offset = torch.tensor([-1] * self.dim + [1] * self.dim, device=deltas.device).unsqueeze(0)
        decoded_bboxes = anchors + offset * deltas

        return decoded_bboxes


if __name__ == "__main__":
    import numpy as np

    anchors = np.array([[1.0, 1.0, 5, 5],
                        [1.0, 1.0, 6, 6],
                        [1.0, 1.0, 2, 2],
                        [1.0, 1.0, 3, 3]])
    targets = np.array([[1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4]])

    d = DeltaBBoxCoder(dim=2, target_means=[0, 0, 0, 0], target_stds=[1., 1., 1., 1.])
    deltas = d.encode(torch.tensor(anchors), torch.tensor(targets))
    box = d.decode(torch.tensor(anchors), deltas)
    print(deltas)
    print(box)

    points = np.array([[1.0, 2.0, 1.0, 2.0],
                       [2.0, 2.2, 2.0, 2.2],
                       [1.0, 2.2, 1.0, 2.2],
                       [2.0, 2.0, 2.0, 2.0]])
    targets = np.array([[1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4]])
    d = DistBBoxCoder(dim=2, target_means=[0, 0, 0, 0], target_stds=[1., 1., 1., 1.])
    deltas = d.encode(torch.tensor(points), torch.tensor(targets))
    box = d.decode(torch.tensor(points), deltas)
    print(deltas)
    print(box)
