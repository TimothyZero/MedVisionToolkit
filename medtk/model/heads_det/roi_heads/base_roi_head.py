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
from abc import abstractmethod

from medtk.model.nnModules import ComponentModule


class BaseRoIHead(ComponentModule):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None):
        super(BaseRoIHead, self).__init__()
    #     if bbox_head is not None:
    #         self.init_bbox_head(bbox_roi_extractor, bbox_head)
    #
    #     if mask_head is not None:
    #         self.init_mask_head(mask_roi_extractor, mask_head)
    #
    # @property
    # def with_bbox(self):
    #     """bool: whether the RoI head contains a `bbox_head`"""
    #     return hasattr(self, 'bbox_head') and self.bbox_head is not None
    #
    # @property
    # def with_mask(self):
    #     """bool: whether the RoI head contains a `mask_head`"""
    #     return hasattr(self, 'mask_head') and self.mask_head is not None
    #
    # @property
    # def with_shared_head(self):
    #     """bool: whether the RoI head contains a `shared_head`"""
    #     return hasattr(self, 'shared_head') and self.shared_head is not None
    #
    # @abstractmethod
    # def init_weights(self):
    #     """Initialize the weights in head."""
    #     pass
    #
    # @abstractmethod
    # def init_bbox_head(self, bbox_roi_extractor, bbox_head):
    #     """Initialize ``bbox_head``"""
    #     pass
    #
    # @abstractmethod
    # def init_mask_head(self, mask_roi_extractor, mask_head):
    #     """Initialize ``mask_head``"""
    #     pass
