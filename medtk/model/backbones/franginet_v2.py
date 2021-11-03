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
from itertools import combinations_with_replacement

from medtk.model.nd import AvgPoolNd
from medtk.model.nnModules import ComponentModule


def _sort_by_abs(a, axis=0):
    """Sort array along a given axis by the absolute value
    modified from: http://stackoverflow.com/a/11253931/4067734
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = torch.abs(a).argsort(axis)
    return a[tuple(index)]


def initHessianFilter(dim, sigma, truncate=3.0):
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

    g = ndi.gaussian_filter(img, sigma, mode='constant')

    axes = range(dim)
    gradients = np.gradient(g)
    # 2d Dxx, Dxy, Dyy
    # 3d Dxx, Dxy, Dxz, Dyy, Dyz, Dzz
    Ds = [np.gradient(gradients[ax0], axis=ax1).astype(np.float32)
          for ax0, ax1 in combinations_with_replacement(axes, 2)]

    # shape of 2d : 3, ...
    # shape of 3d : 6, ...
    params = np.array([[D * (sigma ** 2)] for D in Ds])
    return g, kernel_size, params


def getEigenValues(hessian_matrix: List[List[torch.Tensor]], device):
    """
    :param hessian_matrix:
        [ gxx, gxy, gxz]    |      [ gxx, gxy ]
        [ gyx, gyy, gyz]    |      [ gyx, gyy ]
        [ gzx, gzy, gzz]    |
    :param device: device
    :return: eigenvalues
    """
    hessian_rows = list()
    for row in hessian_matrix:
        hessian_rows.append(torch.stack(row, dim=-1))

    hessian = torch.stack(hessian_rows, dim=-2)
    eigenvalues, _ = torch.symeig(hessian.float(), eigenvectors=True)  # seems not to support CUDA, requires CPU Tensor
    sorted_eigenvalues = _sort_by_abs(eigenvalues, -1)

    # to original device
    eigenvalues = [eigenvalue.squeeze(-1).to(device)
                   for eigenvalue in torch.split(sorted_eigenvalues, [1] * sorted_eigenvalues.shape[-1], dim=-1)]

    return eigenvalues


def Frangi2d(out, betas, isDark):
    Dxx = out[:, [0], ...]
    Dxy = out[:, [1], ...]
    Dyy = out[:, [2], ...]

    hessian_matrix = [[Dxx.cpu(), Dxy.cpu()],
                      [Dxy.cpu(), Dyy.cpu()]]
    eig1, eig2 = getEigenValues(hessian_matrix, Dxx.device)

    Rb2 = torch.div(torch.pow(eig1, 2), torch.pow(eig2, 2) + 1e-6)
    S2 = torch.pow(eig1, 2) + torch.pow(eig2, 2)

    blob = torch.exp(-Rb2 / (2 * torch.pow(betas[-2], 2)))
    background = torch.sub(1, torch.exp(-S2 / (2 * torch.pow(betas[-1], 2))))

    vessel = blob * background
    if isDark:
        vessel = torch.where(eig2 < 0, torch.zeros_like(vessel), vessel)
    else:
        vessel = torch.where(eig2 > 0, torch.zeros_like(vessel), vessel)
    vessel = torch.where(torch.isnan(vessel), torch.zeros_like(vessel), vessel)
    return vessel


def Frangi3d(out, betas, isDark):
    Dxx = out[:, [0], ...]
    Dxy = out[:, [1], ...]
    Dxz = out[:, [2], ...]
    Dyy = out[:, [3], ...]
    Dyz = out[:, [4], ...]
    Dzz = out[:, [5], ...]

    hessian_matrix = [[Dxx.cpu(), Dxy.cpu(), Dxz.cpu()],
                      [Dxy.cpu(), Dyy.cpu(), Dyz.cpu()],
                      [Dxz.cpu(), Dyz.cpu(), Dzz.cpu()]]
    eig1, eig2, eig3 = getEigenValues(hessian_matrix, Dxx.device)

    """
    RA - plate-like structures
    RB - blob-like structures
    S - background
    """
    Ra2 = torch.div(torch.pow(eig2, 2), torch.pow(eig3, 2) + 1e-6)
    Rb2 = torch.div(torch.pow(eig1, 2), torch.abs(torch.mul(eig2, eig3)))
    S2 = torch.pow(eig1, 2) + torch.pow(eig2, 2) + torch.pow(eig3, 2)

    plate = torch.sub(1, torch.exp(-Ra2 / (2 * torch.pow(betas[0], 2))))
    blob = torch.exp(-Rb2 / (2 * torch.pow(betas[1], 2)))
    background = torch.sub(1, torch.exp(-S2 / (2 * torch.pow(betas[2], 2))))

    vessel = plate * blob * background
    if isDark:
        vessel = torch.where(((eig2 < 0) | (eig3 < 0)), torch.zeros_like(vessel), vessel)
    else:
        vessel = torch.where(((eig2 > 0) | (eig3 > 0)), torch.zeros_like(vessel), vessel)
    vessel = torch.where(torch.isnan(vessel), torch.zeros_like(vessel), vessel)
    return vessel


class FrangiNetV2(ComponentModule):
    def __init__(self,
                 dim,
                 in_channels,
                 sigmas,
                 betas,
                 isDark):
        super(FrangiNetV2, self).__init__()
        # dim=2, betas=(beta, c)  |  dim=3, betas=(alpha, beta, c)
        assert len(betas) == dim

        self.dim = dim
        self.in_channels = in_channels
        m_sigma, M_sigma, num = sigmas
        self.sigma_list = np.linspace(m_sigma, M_sigma, num)
        self.betas = torch.nn.Parameter(torch.tensor(betas), requires_grad=True)
        self.isDark = isDark

        self.num_filters = int(self.dim * (self.dim + 1) / 2)  # dim=3, out=6=1+2+3  |  dim=2, out=3=1+2

        self._init_layers()

    def _init_layers(self):
        multi_scale_conv = []
        for i in range(len(self.sigma_list)):
            sigma = self.sigma_list[i]
            g, kernel_size, params = initHessianFilter(self.dim, sigma)
            conv = self.build_conv(self.dim, self.in_channels, self.num_filters,
                                   kernel_size=kernel_size,
                                   padding=int(kernel_size // 2),
                                   bias=False)
            params = np.concatenate([params] * self.in_channels, axis=1)
            assert conv.weight.shape == params.shape, f'conv.weight is {conv.weight.shape} but params is {params.shape}'

            with torch.no_grad():
                conv.weight.data = torch.nn.Parameter(torch.from_numpy(params), requires_grad=True)

            multi_scale_conv.append(conv)

        self.multi_scale_conv = nn.ModuleList(multi_scale_conv)

        out_channels = 3
        mid_channels = 3

        self.out_conv = nn.Sequential(
            self.build_conv(self.dim, out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            self.build_norm(self.dim, mid_channels),
            self.build_act(),
            self.build_conv(self.dim, mid_channels, mid_channels, kernel_size=1, padding=0, bias=False),
        )
        self.dn = AvgPoolNd(self.dim)(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward_single_scale(self, x, scale_index):
        out = self.multi_scale_conv[scale_index](x)
        # print(out.shape)

        if self.dim == 2:
            vessel = Frangi2d(out, self.betas, self.isDark)
        else:
            vessel = Frangi3d(out, self.betas, self.isDark)
        # print(vessel.shape)
        return vessel

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
            if torch.isnan(torch.max(vessel_1)): print("vessel_1 NAN !!! ")
            if torch.isnan(torch.max(vessel_2)): print("vessel_2 NAN !!! ")
            if torch.isnan(torch.max(vessel_3)): print("vessel_3 NAN !!! ")
            outs_1.append(vessel_1)
            outs_2.append(vessel_2)
            outs_3.append(vessel_3)
            # max_out = torch.max(max_out, torch.max(vessel_1, torch.max(vessel_2, vessel_3)))

        outs_1 = torch.max(torch.cat(outs_1, dim=1), dim=1, keepdim=True)[0]
        outs_2 = torch.max(torch.cat(outs_2, dim=1), dim=1, keepdim=True)[0]
        outs_3 = torch.max(torch.cat(outs_3, dim=1), dim=1, keepdim=True)[0]
        stacked = torch.cat([outs_1, outs_2, outs_3], dim=1)

        if torch.isnan(torch.max(stacked)): print("stacked")
        print(torch.max(stacked))

        # print('stacked', stacked.shape)
        outs = self.out_conv(stacked) + stacked
        if torch.isnan(torch.max(outs)): print("outs")
        print(torch.max(outs))
        # print('outs', outs.shape)

        return [outs]
