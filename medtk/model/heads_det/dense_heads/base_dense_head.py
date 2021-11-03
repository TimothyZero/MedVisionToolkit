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
from typing import List
import torch
from medtk.model.nnModules import ComponentModule


class BaseDenseHead(ComponentModule):
    """Base class for DenseHeads."""

    def __init__(self, *args, **kwargs):
        super(BaseDenseHead, self).__init__(*args, **kwargs)

    def get_losses(self, **kwargs):
        """Compute losses of the head."""
        pass

    def get_bboxes(self, *args, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def get_proposals(self, *args, **kwargs):
        pass

    def forward_train(self,
                      feats: List[torch.Tensor],
                      batch_gt_labels,
                      batch_gt_bboxes) -> tuple:
        """

        Args:
            feats: [(B, C, **shape1), (B, C, **shape2), ..]
            batch_gt_labels: (B, N, Coords)
            batch_gt_bboxes: (B, N, Classes)

        Returns:
            if train:
                return losses, None
            if valid:
                return losses, bboxes
            if infer:
                return None, bboxes

        """
        raise NotImplementedError

    def forward_valid(self,
                      feats: List[torch.Tensor],
                      batch_gt_labels,
                      batch_gt_bboxes) -> tuple:
        raise NotImplementedError

    def forward_infer(self, feats: List[torch.Tensor]) -> tuple:
        raise NotImplementedError