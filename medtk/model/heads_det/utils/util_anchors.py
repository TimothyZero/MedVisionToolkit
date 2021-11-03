# -*- coding: utf-8 -*-

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

from typing import Union
import numpy as np
import torch
import torch.nn.functional as F

from medtk.model.nnModules import ComponentModule


class AnchorGenerator(ComponentModule):
    def __init__(self, dim,
                 base_scales: float,
                 scales: Union[tuple, list],
                 ratios: Union[tuple, list],
                 strides: Union[tuple, list] = (2, 4, 8),
                 invalid_margin=0):
        """

        Args:
            dim: dimension of image, 2 or 3
            base_scales: change anchors base size,
                default should be 1, which means using 'stride' as base width;
                if base scales is larger than 1, it will generate larger anchors at same feature
            scales: anchor scales per pixel in feature map
            ratios: anchor ratios per pixel in feature map
            strides:
        """
        self.dim = dim
        self.base_scales = base_scales
        self.scales = scales
        self.ratios = ratios
        self.strides = strides
        self.patch_margin = invalid_margin
        self.num_levels = len(strides)
        assert dim in (2, 3)
        assert (dim == 3) * len(ratios) in (0, 1)
        super(AnchorGenerator, self).__init__()
        # self.info('using base anchors:\n',
        #           self.base_scales * generate_base_anchors_Nd(self.dim, self.scales, self.ratios))
        # self.info('using base strides:\n', self.strides)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '('
        repr_str += f'base_scales={self.base_scales}, scales={self.scales}, '
        repr_str += f'ratios={self.ratios}, strides={self.strides}, '
        repr_str += f'patch_margin={self.patch_margin}'
        repr_str += ')'
        return repr_str

    @property
    def num_base_anchors(self):
        return len(self.scales) * len(self.ratios)

    @staticmethod
    def generate_base_anchors_Nd(
            dim: int,
            scales: Union[tuple, list],
            ratios: Union[tuple, list] = None) -> np.ndarray:
        """
        :param dim:
        :param scales:
        :param ratios:
        :return: base_anchors: numpy array  [n,(x1,y1,z1,x2,y2,z2)] or [n,(x1,y1,x2,y2)]
        """
        scales = np.array(scales, np.float)
        # [nb : num of base_anchors]
        if dim == 2:
            ratios = np.array(ratios, np.float)
            h_ratios = np.sqrt(ratios)
            w_ratios = 1 / h_ratios
            ws = (w_ratios[:, None] * scales[None, :]).flatten()
            hs = (h_ratios[:, None] * scales[None, :]).flatten()
            # [nb,(y1,x1,y2,x2)]
            base_anchors = np.stack([- 0.5 * ws, - 0.5 * hs,
                                     0.5 * ws, 0.5 * hs], axis=1)
        else:
            # [nb,(x1,y1,z1,x2,y2,z2)]
            base_anchors = np.stack([-0.5 * scales, -0.5 * scales, -0.5 * scales,
                                     0.5 * scales, 0.5 * scales, 0.5 * scales], axis=1)
        return base_anchors

    @staticmethod
    def shift_Nd(
            dim: int,
            base_anchors: np.ndarray,
            stride: int,
            feature_shape) -> np.ndarray:
        """
        :param dim
        :param base_anchors: numpy [n,(x1,y1,z1,x2,y2,z2)]
        :param stride:
        :param feature_shape: feature_shape
        :return: anchors: [n * 6(x1,y1,z1,x2,y2,z2), d, h, w]
        """
        # I think:
        # in python: the spatial coord of the center of first pixel is (0., 0.),
        # so the top left coord is (-0.5, -0.5)
        ctr_Nd = [np.arange(0, s) * stride + 0.5 * stride - 0.5 for s in feature_shape]  # zyx order

        # pay attention to indexing arg : indexing='ij'
        # each shape is [(d), h, w]
        ctr_Nd = np.meshgrid(*ctr_Nd, indexing='ij')
        # xyz order [6(x1,y1,z1,x2,y2,z2), d, h, w]
        ctr_Nd = np.concatenate([ctr_Nd[::-1], ctr_Nd[::-1]], axis=0)
        # [1, 6, d, h, w]
        ctr_Nd = np.expand_dims(ctr_Nd, axis=0)
        # [nb, 6, 1, 1, 1]
        base_anchors = np.expand_dims(base_anchors, axis=tuple(range(2, dim + 2)))
        # [nb, 6(x1,y1,z1,x2,y2,z2), d, h, w]
        anchors = ctr_Nd + base_anchors
        # [n = nb * 6(x1,y1,z1,x2,y2,z2), d, h, w]
        anchors = np.reshape(anchors, [-1] + list(feature_shape))
        return anchors

    def forward_single_level(self, feature_size, pixel_stride):
        """

        Args:
            feature_size (tuple): [h, w] or [d, h, w]
            pixel_stride (int): equals to raw image shape // feature map shape

        Returns:
            torch.Tensor:
                anchors: Tensor [Num, (d,) h, w]
        """
        # nb = num_base_anchors
        # [n = nb * 4(x1,y1,x2,y2), h, w]
        # or [n = nb * 6(x1,y1,z1,x2,y2,z2), d, h, w]
        base_anchors = self.base_scales * pixel_stride * self.generate_base_anchors_Nd(self.dim, self.scales, self.ratios)
        shifted_anchors = self.shift_Nd(self.dim, base_anchors, pixel_stride, feature_size)
        anchors = torch.from_numpy(shifted_anchors.astype(np.float32)).float()

        margin_on_feature = self.patch_margin // pixel_stride
        valid_shape = torch.tensor(anchors.shape[1:]) - 2 * margin_on_feature
        valid_index = torch.ones(anchors.shape[0], *valid_shape, device=anchors.device)
        valid_index = F.pad(valid_index, [margin_on_feature, ] * self.dim * 2)
        valid_index = valid_index.bool()

        return anchors, valid_index

    def forward(self, feature_sizes, batch_size=1, device='cuda'):
        """

        Args:
            feature_sizes (list[tuple]):
            batch_size: int
            device:

        Returns:
            tuple:
                - multi_level_anchors (list[tensor]):
                - num_of_anchors (list[int]):

        """
        assert len(feature_sizes) == len(self.strides)
        multi_level_anchors, multi_level_valid, num_level_anchors = [], [], []
        for feature_shape, pixel_stride in zip(feature_sizes, self.strides):
            anchors, valid_index = self.forward_single_level(feature_shape, pixel_stride)

            anchors = torch.stack([anchors] * batch_size)
            valid_index = torch.stack([valid_index] * batch_size)

            multi_level_anchors.append(anchors.to(device))
            multi_level_valid.append(valid_index.to(device))
            num_level_anchors.append(anchors.numel() // (2 * self.dim) // batch_size)

        return multi_level_anchors, multi_level_valid, num_level_anchors

    def forward_previous(self, feature_sizes, batch_size=1, device='cuda'):
        """

        Args:
            feature_sizes (list[tuple]):
            device:

        Returns:
            tuple:
                - multi_level_anchors (list[tensor]):
                - num_of_anchors (list[int]):

        """
        assert len(feature_sizes) == len(self.strides)
        multi_level_anchors, multi_level_valid, num_level_anchors = [], [], []
        for feature_shape, pixel_stride in zip(feature_sizes, self.strides):
            anchors, valid_index = self.forward_single_level(feature_shape, pixel_stride)

            axes = list(range(1, self.dim + 1)) + [0]
            anchors = anchors.permute(*axes).contiguous().view(-1, 2 * self.dim)
            valid_index = valid_index.permute(*axes).contiguous().view(-1, 2 * self.dim)

            multi_level_anchors.append(anchors.to(device))
            multi_level_valid.append(valid_index.to(device))
            num_level_anchors.append(anchors.numel() // (2 * self.dim))

        return multi_level_anchors, multi_level_valid, num_level_anchors


class PointGenerator(ComponentModule):
    def __init__(self, dim,
                 strides: Union[tuple, list] = (2, 4, 8),
                 invalid_margin=0):
        """

        Args:
            dim: dimension of image, 2 or 3
            strides:
        """
        self.dim = dim
        self.strides = strides
        self.patch_margin = invalid_margin
        self.num_levels = len(strides)
        assert dim in (2, 3)
        super(PointGenerator, self).__init__()
        # self.info('using base anchors:\n',
        #           self.base_scales * generate_base_anchors_Nd(self.dim, self.scales, self.ratios))
        # self.info('using base strides:\n', self.strides)

    @property
    def num_base_anchors(self):
        return 1

    @staticmethod
    def generate_points_Nd(
            dim: int,
            stride: int,
            feature_shape) -> np.ndarray:
        """
        :param dim
        :param stride:
        :param feature_shape: feature_shape
        :return: anchors: [n * 6(x1,y1,z1,x2,y2,z2), d, h, w]
        """
        # I think:
        # in python: the spatial coord of the center of first pixel is (0., 0.),
        # so the top left coord is (-0.5, -0.5)
        ctr_Nd = [np.arange(0, s) * stride + 0.5 * stride - 0.5 for s in feature_shape]  # zyx order

        # pay attention to indexing arg : indexing='ij'
        # each shape is [(d), h, w]
        ctr_Nd = np.meshgrid(*ctr_Nd, indexing='ij')
        # xyz order [6(x1,y1,z1,x2,y2,z2), d, h, w]
        anchors = np.concatenate([ctr_Nd[::-1], ctr_Nd[::-1]], axis=0)
        return anchors

    def forward_single_level(self, feature_size, pixel_stride):
        # [6(x1,y1,z1,x2,y2,z2), d, h, w]
        points = self.generate_points_Nd(self.dim, pixel_stride, feature_size)
        points = torch.from_numpy(points.astype(np.float32)).float()

        margin_on_feature = self.patch_margin // pixel_stride
        valid_shape = torch.tensor(points.shape[1:]) - 2 * margin_on_feature
        valid_index = torch.ones(points.shape[0], *valid_shape, device=points.device)
        valid_index = F.pad(valid_index, [margin_on_feature, ] * self.dim * 2)
        valid_index = valid_index.bool()

        return points, valid_index

    def forward(self, feature_sizes, batch_size=1,  device='cuda'):
        """

        Args:
            feature_sizes (list[tuple]):
            batch_size: int
            device:

        Returns:
            tuple:
                - multi_level_anchors (list[tensor]):
                - num_of_anchors (list[int]):

        """
        assert len(feature_sizes) == len(self.strides)
        multi_level_anchors, multi_level_valid, num_level_anchors = [], [], []
        for feature_shape, pixel_stride in zip(feature_sizes, self.strides):
            anchors, valid_index = self.forward_single_level(feature_shape, pixel_stride)

            anchors = torch.stack([anchors] * batch_size)
            valid_index = torch.stack([valid_index] * batch_size)

            multi_level_anchors.append(anchors.to(device))
            multi_level_valid.append(valid_index.to(device))
            num_level_anchors.append(anchors.numel() // (2 * self.dim) // batch_size)

        return multi_level_anchors, multi_level_valid, num_level_anchors


if __name__ == '__main__':
    # 2d
    features = [torch.zeros((1, 1, 24, 24)), torch.zeros((1, 1, 12, 12))]
    anchorsNet = AnchorGenerator(dim=2, base_scales=1, scales=[1], ratios=[1], strides=[4, 8])

    # feature_shape = feature.shape[2:]
    # multi_level_anchors, multi_level_valid, num_level_anchors = anchorsNet.forward_single_level(feature_shape, pixel_stride=4)
    # print(multi_level_anchors[0][0, :, 0, 0], '-->', multi_level_valid[0][0, :, 0, 0])
    # print(multi_level_anchors[0][0, :, 0, 1], '-->', multi_level_valid[0][0, :, 0, 1])
    # print(multi_level_anchors[0][0, :, 1, 0], '-->', multi_level_valid[0][0, :, 1, 0])
    # print(multi_level_anchors[0][0, :, 1, 1], '-->', multi_level_valid[0][0, :, 1, 1])
    # print(multi_level_anchors[0][0, :, 2, 1], '-->', multi_level_valid[0][0, :, 2, 1])
    # print(multi_level_anchors[0][0, :, 3, 1], '-->', multi_level_valid[0][0, :, 3, 1])

    multi_level_anchors, multi_level_valid, num_level_anchors = anchorsNet.forward_previous([f.shape[2:] for f in features], device='cpu')
    print(num_level_anchors)

