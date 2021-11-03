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

from typing import List
import numpy as np
import scipy.ndimage as ndi
import torch
from torch import nn

from medtk.model.nd import AvgPoolNd
from medtk.model.nnModules import ComponentModule


def initHessianFilter(dim, sigma, truncate=3.0, sqrt=3.0):
    """
    initial conv kernels with Hessian Kernel
    :param dim: 2 or 3
    :param sigma: gaussian sigma, will match [2 * sigma] width vessel
    :param truncate: kernel size
    :return:
    """
    sigma = float(sigma)
    half_win = int(truncate * sigma + 0.5)
    kernel_size = 2 * half_win + 1

    # fill pixel or voxel at center [half_win, half_win] or [half_win, half_win, half_win]
    img = np.zeros([kernel_size] * dim)
    img[tuple([half_win] * dim)] = 1

    if dim == 2:
        sigmas = [sigma / np.power(sqrt, 0.5), sigma * np.power(sqrt, 0.5)]
    else:
        sigmas = [sigma / np.power(sqrt, 0.25), sigma / np.power(sqrt, 0.25), sigma * np.power(sqrt, 0.5)]
    g = ndi.gaussian_filter(img, sigmas, mode='constant')

    gradients = np.gradient(g)
    # 2d Dxx, Dxy, Dyy
    # 3d Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
    if dim == 2:
        Dxx, Dxy = np.gradient(gradients[0])
        V = Dxx
        B = Dxy
        Ds = [V,
              ndi.rotate(V, 45, axes=[0, 1], reshape=False),
              ndi.rotate(V, 90, axes=[0, 1], reshape=False),
              ndi.rotate(V, 135, axes=[0, 1], reshape=False)]
    else:
        Dxx, Dxy, Dxz = np.gradient(gradients[0])
        Dyx, Dyy, Dyz = np.gradient(gradients[1])
        V = Dxx / 2 + Dyy / 2
        B = Dxz / 2 + Dyz / 2
        Ds = [V,
              ndi.rotate(V, 90, axes=[0, 2], reshape=False),
              ndi.rotate(V, 90, axes=[2, 1], reshape=False),

              ndi.rotate(V, 45, axes=[0, 2], reshape=False),
              ndi.rotate(V, 135, axes=[0, 2], reshape=False),

              ndi.rotate(V, 45, axes=[2, 1], reshape=False),
              ndi.rotate(V, 135, axes=[2, 1], reshape=False),

              ndi.rotate(ndi.rotate(V, 90, axes=[0, 2], reshape=False), 45, axes=[0, 1], reshape=False),
              ndi.rotate(ndi.rotate(V, 90, axes=[0, 2], reshape=False), 135, axes=[0, 1], reshape=False),

              #              ndi.rotate(ndi.rotate(V, 45, axes=[0, 2], reshape=False), 45, axes=[0, 1], reshape=False),
              #               ndi.rotate(ndi.rotate(V, 45, axes=[0, 2], reshape=False), 135, axes=[0, 1], reshape=False),

              #               ndi.rotate(ndi.rotate(V, 45, axes=[0, 2], reshape=False), 135, axes=[2, 1], reshape=False),
              ]
    Ds = [np.ascontiguousarray(D).astype(np.float32) for D in Ds]

    # shape of 2d : 3, ...
    # shape of 3d : 6, ...
    params = np.array([[D * (sigma ** dim)] for D in Ds])
    return g, kernel_size, params


class FastVFrangiNetV2(ComponentModule):
    def __init__(self,
                 dim,
                 in_channels,
                 sigmas,
                 betas,
                 isDark):
        super(FastVFrangiNetV2, self).__init__()
        # dim=2, betas=(beta, c)  |  dim=3, betas=(alpha, beta, c)
        assert len(betas) == dim

        self.dim = dim
        self.in_channels = in_channels
        m_sigma, M_sigma, num = sigmas
        self.sigma_list = np.linspace(m_sigma, M_sigma, num)
        self.betas = torch.nn.Parameter(torch.tensor(betas), requires_grad=True)
        self.isDark = isDark

        self.num_filters = self.dim ** 2

        self._init_layers()

    def _init_layers(self):
        multi_scale_conv = []
        for i in range(len(self.sigma_list)):
            sigma = self.sigma_list[i]
            g, kernel_size, params = initHessianFilter(self.dim, sigma)
            conv = self.build_conv(self.dim, self.in_channels, self.num_filters,
                                   kernel_size=kernel_size,
                                   padding=int(kernel_size // 2),
                                   padding_mode='zeros',
                                   bias=False)
            params = np.concatenate([params] * self.in_channels, axis=1)
            assert conv.weight.shape == params.shape, f'conv.weight is {conv.weight.shape} but params is {params.shape}'

            with torch.no_grad():
                conv.weight.data = torch.nn.Parameter(torch.from_numpy(params), requires_grad=True)

            multi_scale_conv.append(nn.Sequential(
                conv,
                self.build_norm(self.dim, self.num_filters),
                self.build_act()
            ))

        self.multi_scale_conv = nn.ModuleList(multi_scale_conv)

        out_channels = 3 * len(self.sigma_list) * self.num_filters
        mid_channels = 60

        self.out_conv = nn.Sequential(
            self.build_conv(self.dim, out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            self.build_norm(self.dim, mid_channels),
            self.build_act(),
            self.build_conv(self.dim, mid_channels, 32, kernel_size=1, padding=0, bias=False),
        )
        self.dn = AvgPoolNd(self.dim)(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward_single_scale(self, x, scale_index):
        out = self.multi_scale_conv[scale_index](x)
        # print(out.shape)
        return out

    def forward(self, img):
        """
        :param img: [b, c, h, w, (d)]
        :return:
        """
        assert torch.max(img) <= 1.0, f'max of img is {torch.max(img).item()}'
        assert np.min(img.shape[2:]) >= 16

        outs_1 = []
        outs_2 = []
        outs_3 = []
        for i in range(len(self.sigma_list)):
            vessel_1 = self.forward_single_scale(128.0 + img * 128.0, i)
            vessel_2 = self.up(self.forward_single_scale(self.dn(128.0 + img * 128.0), i))
            vessel_3 = self.up(self.up(self.forward_single_scale(self.dn(self.dn(128.0 + img * 128.0)), i)))
            if torch.isnan(torch.max(vessel_1)): print("vessel_1")
            if torch.isnan(torch.max(vessel_2)): print("vessel_2")
            if torch.isnan(torch.max(vessel_3)): print("vessel_3")
            outs_1.append(vessel_1)
            outs_2.append(vessel_2)
            outs_3.append(vessel_3)
            # max_out = torch.max(max_out, torch.max(vessel_1, torch.max(vessel_2, vessel_3)))

        outs_1 = torch.cat(outs_1, dim=1)
        outs_2 = torch.cat(outs_2, dim=1)
        outs_3 = torch.cat(outs_3, dim=1)
        stacked = torch.cat([outs_1, outs_2, outs_3], dim=1)

        if torch.isnan(torch.max(stacked)): print("stacked")
        # print(torch.max(stacked))

        # print('stacked', stacked.shape)
        outs = self.out_conv(stacked)
        if torch.isnan(torch.max(outs)): print("outs")
        # print(torch.max(outs))
        # print('outs', outs.shape)

        return [outs]
