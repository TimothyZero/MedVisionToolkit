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

from collections.abc import Sequence
from datetime import datetime
import numpy as np
import torch
import time

from .aug_base import Stage


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def to_numpy(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return np.append(data)
    elif isinstance(data, torch.LongTensor):
        return data.numpy()
    # elif isinstance(data, float):
    #     return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class ToTensor(Stage):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self.forward(r) for r in results]
        else:
            return [self.backward(r) for r in results]

    def forward(self, results):
        if self.keys is not None:
            keys = self.keys
        else:
            keys = results.keys()
        for key in keys:
            results[key] = to_tensor(results[key])
        return results

    def backward(self, results):
        if self.keys is not None:
            keys = self.keys
        else:
            keys = results.keys()
        for key in keys:
            results[key] = to_numpy(results[key])
        return results


class ToArray(ToTensor):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def forward(self, results):
        return super().backward(results)

    def backward(self, results):
        return super().forward(results)

# @PIPELINES.register_module
# class ImageToTensor(object):
#
#     def __init__(self, keys):
#         self.keys = keys
#
#     def __call__(self, results):
#         for key in self.keys:
#             results[key] = to_tensor(results[key].transpose(2, 0, 1))
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={})'.format(self.keys)


# @PIPELINES.register_module
# class Transpose(object):
#
#     def __init__(self, keys, order):
#         self.keys = keys
#         self.order = order
#
#     def __call__(self, results):
#         for key in self.keys:
#             results[key] = results[key].transpose(self.order)
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={}, order={})'.format(
#             self.keys, self.order)


# @PIPELINES.register_module
# class DefaultFormatBundle(Stage):
#     """Default formatting bundle.
#
#     It simplifies the pipeline of formatting common fields, including "img",
#     "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
#     These fields are formatted as follows.
#
#     - img:       (1)transpose, (2)to tensor
#     - gt_seg:    (1)transpose, (2)to tensor
#     """
#     def __repr__(self):
#         return self.__class__.__name__
#
#     @property
#     def canBackward(self):
#         return True
#
#     def __call__(self, results, forward=True):
#         if isinstance(results, dict):
#             results = [results]
#
#         if forward:
#             return [self.forward(r.copy()) for r in results]
#         else:
#             return [self.backward(r.copy()) for r in results]
#
#     def forward(self, results):
#         # for key in results['seg_fields']:
#         #     img = results[key]
#         #     results[key] = to_tensor(img)  # DC(to_tensor(img), stack=True)
#         # for key in results['img_fields']:
#         #     img = results[key]
#         #     results[key] = to_tensor(img)  # DC(to_tensor(img), stack=True)
#         # if 'patches_img' in results.keys():
#         #     patches = []
#         #     for patch in results['patches_img']:
#         #         patch = to_tensor(patch)
#         #         patches.append(patch)
#         #     patches = torch.stack(patches, dim=0)
#         #     results['patches_img'] = patches
#         for key in ['gt_det', 'gt_cls']:
#             if key not in results:
#                 continue
#             results[key] = to_tensor(results[key])  # DC(to_tensor(results[key]))
#         results['history'].append(self.name)
#         return results
#
#     def backward(self, results):
#         # for key in results['seg_fields']:
#         #     results[key] = to_numpy(results[key])
#         # for key in results['img_fields']:
#         #     results[key] = to_numpy(results[key])
#         # if 'patches_img' in results.keys():
#         #     patches = []
#         #     for patch in results['patches_img']:
#         #         patch = to_numpy(patch)
#         #         patches.append(patch)
#         #     patches = np.stack(patches, axis=0)
#         #     results['patches_img'] = patches
#         for key in ['proposals', 'gt_det', 'gt_cls']:
#             if key not in results:
#                 continue
#             results[key] = to_numpy(results[key])  # DC(to_tensor(results[key]))
#         last = results['history'].pop()
#         assert last == self.name
#         return results


class Collect(Stage):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self.forward(r.copy()) for r in results]
        else:
            return [self.backward(r.copy()) for r in results]

    def forward(self, results):
        _tic_ = time.time()
        data = {}
        img_meta = {}
        for key in self.keys:
            data[key] = to_tensor(results.pop(key))
        for key in ['patches_img']:
            if key in results.keys():
                data[key] = to_tensor(results.pop(key))
        for key in results.keys():
            img_meta[key] = results[key]
        _start = datetime.fromtimestamp(_tic_).strftime('%H:%M:%S.%f')
        data['img_meta'] = img_meta
        data['img_meta']['history'].append(self.name)
        data['img_meta']['time'].append(f'{self.name}-{_start}-{time.time() - _tic_:.03f}s')

        return data

    def backward(self, data):
        results = {}
        for key in self.keys:
            results[key] = to_numpy(data[key])
        for key in ['patches_img']:
            if key in data.keys():
                results[key] = to_numpy(data[key])
        for k, v in data['img_meta'].items():
            results[k] = v
        time_cost = results['time'].pop()
        last = results['history'].pop()
        assert last == self.name
        return results
