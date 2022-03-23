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

from typing import List, Union
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

from medtk.model.nd import ConvNd, BatchNormNd
from medtk.model.nnModules import ComponentModule
from ..losses import CrossEntropyLoss


class BaseSegHead(ComponentModule):
    def __init__(self,
                 dim: int,
                 in_channels: List[int],
                 num_classes: int,
                 loss_cls: Union[dict, object],
                 metrics: Union[List[dict], List[object]]):
        super(BaseSegHead, self).__init__()
        assert isinstance(in_channels, list), 'must be a list'
        assert isinstance(in_channels, list), 'must be a list'
        metrics = [metrics] if isinstance(metrics, dict) else metrics

        self.dim = dim
        self.in_channels = in_channels
        if getattr(loss_cls, 'use_sigmoid', False):
            self.out_channels = num_classes
        else:
            self.out_channels = num_classes + 1

        self.loss_cls = loss_cls
        self.base_criterion = CrossEntropyLoss()
        self.metrics = nn.ModuleList([metric for metric in metrics])
        self.layers = None
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self):
        layers = []
        for in_channel in self.in_channels:
            layers.append(self._make_single_out(in_channel, self.out_channels))
        self.layers = nn.ModuleList(layers)
    
    def _make_single_out(self, in_channels, out_channels):
        """ seems larger size can achieve higher performance """
        # TODO whether to use bias=false, kernel_size=5
        out = self.build_conv(self.dim, in_channels, out_channels, kernel_size=1, padding=0)
        return out

    def _init_weights(self):
        pass
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         n = np.prod(m.kernel_size) * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif self.is_norm(self.dim, m):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #
        # for layer in self.layers:
        #     layer.weight.data.fill_(0)

    def forward(self, x_nk, x_bk=None):
        assert len(self.in_channels) == len(x_nk), f'backbone outs({len(x_nk)}) not match head ins({len(self.in_channels)})'
        outs = []
        for i in range(len(self.in_channels)):
            outs.append(self.layers[i](x_nk[i]))

        if self.monitor_enable:
            self.monitor_fun(outs)
        return outs

    def monitor_fun(self, outs):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import os

        from medtk.data.visulaize import getSeg2D

        matplotlib.use('agg')

        cur_epoch = self.runner_data['epoch'] + 1
        filename = self.runner_data['img_meta'][0]['filename']
        cur_iter = self.runner_data['iter']
        result_dir = self.runner_data_meta['result_dir']
        os.makedirs(result_dir, exist_ok=True)

        data = self.runner_data

        if data['img'].ndim == 4:
            seg = data['gt_seg'][0].permute(1, 2, 0).detach().cpu().numpy()
            im = data['img'][0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
            im = Normalize()(im)
            if im.shape[-1] not in [1, 3]:
                im = im[..., 0]
            show = getSeg2D(im, seg)
            plt.imshow(show)
        else:
            seg = data['gt_seg'][0].permute(1, 2, 3, 0)
            z_idx = torch.argmax(torch.sum(seg, dim=(1, 2)))
            im = data['img'][0].permute(1, 2, 3, 0).detach().cpu().numpy() * 0.5 + 0.5
            im = Normalize()(im)
            show = getSeg2D(im[z_idx], seg[z_idx].detach().cpu().numpy())
            plt.imshow(show)

        fig = plt.gcf()
        fig.savefig(
            os.path.join(
                result_dir,
                f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewData.jpg"),
            dpi=200)
        plt.close(fig)

        net_output = outs[0].float()
        if data['img'].ndim == 4:
            im = data['img'][0].float().permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
            im = Normalize()(im)

            cls = net_output.shape[1]
            if getattr(self.loss_cls, 'use_sigmoid', False):
                net_output = torch.softmax(net_output, dim=1)
            else:
                net_output = net_output.sigmoid()
            fig, ax = plt.subplots(1, cls + 1, figsize=(5 * cls + 5, 5))
            ax0 = ax[0].imshow(im)
            fig.colorbar(ax0, ax=ax[0])
            for i in range(net_output.shape[1]):
                ax0 = ax[i + 1].imshow(net_output[0, i, ...].detach().cpu().numpy(), vmin=0, vmax=1)
                fig.colorbar(ax0, ax=ax[i + 1])
        else:
            im = data['img'][0].float().permute(1, 2, 3, 0).detach().cpu().numpy() * 0.5 + 0.5
            im = Normalize()(im)

            if 'gt_seg' in data.keys():
                seg = data['gt_seg'][0].permute(1, 2, 3, 0)
                z_idx = torch.argmax(torch.sum(seg, dim=(1, 2)))
            else:
                pred = net_output[0].permute(1, 2, 3, 0)[..., :1]
                z_idx = torch.argmin(pred.view(pred.shape[0], -1).min(dim=1)[0])
                # print(z_idx)

            cls = net_output.shape[1]
            if getattr(self.loss_cls, 'use_sigmoid', False):
                net_output = torch.softmax(net_output, dim=1)
            else:
                net_output = net_output.sigmoid()
            fig, ax = plt.subplots(1, cls + 1, figsize=(5 * cls + 5, 5))
            ax0 = ax[0].imshow(im[z_idx, ..., 0])
            fig.colorbar(ax0, ax=ax[0])
            for i in range(net_output.shape[1]):
                ax0 = ax[i + 1].imshow(net_output[0, i, z_idx, ...].detach().cpu().numpy(), vmin=0, vmax=1)
                fig.colorbar(ax0, ax=ax[i + 1])
        fig = plt.gcf()
        fig.savefig(
            os.path.join(
                result_dir,
                f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewFeatures.jpg"),
            dpi=200)
        plt.close(fig)

        if self.runner_data_meta['mode'] == 'valid':
            if data['img'].ndim == 4:
                im = data['img'][0].float().permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
                im = Normalize()(im)

                if im.shape[-1] not in [1, 3]:
                    im = im[..., 0]
                if net_output.shape[1] == 1:
                    seg = net_output.sigmoid()
                else:
                    cls = net_output.shape[1]
                    seg = torch.true_divide(torch.argmax(net_output, dim=1), cls - 1)
                seg = seg[0].detach().cpu().numpy()
                show = getSeg2D(im, seg)
                plt.imshow(show)
            else:
                im = data['img'][0].float().permute(1, 2, 3, 0).detach().cpu().numpy() * 0.5 + 0.5
                im = Normalize()(im)

                if 'gt_seg' in data.keys():
                    seg = data['gt_seg'][0].permute(1, 2, 3, 0)
                    z_idx = torch.argmax(torch.sum(seg, dim=(1, 2)))
                else:
                    pred = net_output[0].permute(1, 2, 3, 0)[..., :1]
                    z_idx = torch.argmin(pred.view(pred.shape[0], -1).min(dim=1)[0])

                cls = net_output.shape[1]
                seg = torch.true_divide(torch.argmax(net_output, dim=1), cls - 1)
                seg = seg[0].detach().cpu().numpy()
                show = getSeg2D(im[z_idx], seg[z_idx])
                plt.imshow(show)
            fig = plt.gcf()
            fig.savefig(
                os.path.join(
                    result_dir,
                    f"Epoch{cur_epoch}_Iter{cur_iter}_{filename}_viewDataResults.jpg"),
                dpi=200)
            plt.close(fig)

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output, ground_truth)
            metrics.update(one_metric)
        return metrics

    def loss(self, net_output, ground_truth):
        losses = self.loss_cls(net_output, ground_truth)
        if isinstance(losses, torch.Tensor):
            losses = {'loss': losses}
        return losses

    def loss_reference(self, net_output, ground_truth):
        return self.base_criterion(net_output, ground_truth)

    def forward_train(self, net_outputs, gt_seg):
        if gt_seg.dim() == self.dim + 2:
            gt_seg = gt_seg.squeeze(1)

        rescale = 2 * (1 - 0.5 ** len(net_outputs))
        # print(rescale, len(net_output))
        losses = OrderedDict()
        for i in range(len(net_outputs)):
            prediction = net_outputs[i]
            if prediction.shape[2:] != gt_seg.shape[1:]:
                scale = int(gt_seg.shape[1] / prediction.shape[2])
                prediction = F.interpolate(prediction, scale_factor=scale)

            loss_dict = self.loss(prediction, gt_seg.long())
            reference = self.loss_reference(prediction, gt_seg.long())
            # print(loss_dict, reference)
            assert isinstance(loss_dict, dict), "head.loss must return a dict of losses"
            assert isinstance(reference, torch.Tensor), "head.base_loss must return a tensor"

            losses['reference'] = losses.get('reference', 0) + reference * (0.5 ** i) / rescale
            for loss_name, loss_value in loss_dict.items():
                losses[loss_name] = losses.get(loss_name, 0) + loss_value * (0.5 ** i) / rescale
        return losses, None

    def forward_valid(self, outs, gt_seg):
        losses = self.loss(outs, gt_seg)
        return losses, outs

    def forward_infer(self, outs):
        return None, outs


class VBNetOutput(nn.Module):
    def __init__(self, in_channels, num_classes, loss_cls):
        super(VBNetOutput, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'must be a list/tuple'
        self.in_channels = in_channels
        self.out_channels = num_classes + 1
        convs = []
        for in_channel in in_channels:
            convs.append(
                nn.Sequential(
                    # nn.Conv3d(in_channel, self.out_channels, kernel_size=3, padding=1),
                    # nn.BatchNorm3d(self.out_channels),
                    # nn.ReLU(inplace=True),
                    nn.Conv3d(in_channel, self.out_channels, kernel_size=1),
                    # nn.Softmax(dim=1)
                ))
        self.convs = nn.ModuleList(convs)

        self.criterion = loss_cls
        self.base_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        assert len(self.in_channels) == len(x), 'backbone/neck outs not match head ins'
        outs = []
        for i in range(len(self.in_channels)):
            outs.append(self.convs[i](x[i]))
        return outs

    @staticmethod
    def to_results(outs):
        if isinstance(outs, list):
            results = []
            for out in outs:
                results.append(out.softmax(dim=1)[:, 1:])
        else:
            results = outs.softmax(dim=1)[:, 1:]
        return results

    def loss(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)

    def loss_reference(self, prediction, ground_truth):
        return self.base_criterion(prediction, ground_truth)


class PassThrough(nn.Module):
    def __init__(self, dim, in_channels, num_classes, loss_cls, metrics):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = num_classes

        self.criterion = loss_cls
        self.metrics = nn.ModuleList(metrics)
        self.base_criterion = CrossEntropyLoss()

    def forward(self, x_nk, x_bk=None):
        return x_nk

    @staticmethod
    def to_results(outs):
        if isinstance(outs, list):
            results = []
            for out in outs:
                results.append(out.softmax(dim=1)[:, 1:])
        else:
            results = outs.softmax(dim=1)[:, 1:]
        return results

    def metric(self, net_output, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(net_output, ground_truth)
            metrics.update(one_metric)
        return metrics

    def loss(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)

    def loss_reference(self, prediction, ground_truth):
        return self.base_criterion(prediction, ground_truth)
