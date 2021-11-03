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

import platform
import numpy as np
import random
import torch
from functools import partial

from torch.utils.data import DataLoader

from ..parallel.dist_utils import get_dist_info
from .sampler import DistributedSampler
from .cudadataloader import CudaDataLoader

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate(batch_data_list):
    # a list of list
    import time

    _tic_ = time.time()

    if isinstance(batch_data_list[0], list):
        # assert all([len(i) == 1 and type(i[0]) == dict for i in batch_data_list]), 'All data should only contains one dict'
        # if not all([len(i) == 1 and type(i[0]) == dict for i in data_list]):  # 'All data should only contains one dict'
        #     assert len(data_list) == 1  # if data contains multi dict
        #     data_list = [[i] for i in data_list[0]]
        batch_data_list = [j for i in batch_data_list for j in i]

    try:
        batch = {'img_meta': [s['img_meta'] for s in batch_data_list]}
        for key in batch_data_list[0].keys():
            if key != 'img_meta' and 'img' in key:
                batch[key] = torch.stack([s[key] for s in batch_data_list], dim=0)
    except Exception as e:
        [print(s['img_meta']) for s in batch_data_list]
        [print(s['img'].shape, s['img'].dtype) for s in batch_data_list]
        raise e

    if 'gt_det' in batch_data_list[0].keys():
        dim = batch_data_list[0]['img_meta']['img_dim']
        gt_bboxes = [s['gt_det'] for s in batch_data_list]

        max_num_labels = max(b.shape[0] for b in gt_bboxes)
        min_num_labels = min(b.shape[0] for b in gt_bboxes)
        # print(dim, min_num_labels, max_num_labels)

        # if max_num_labels > 0:
        gt_bboxes_pad = torch.ones((len(gt_bboxes), max_num_labels, 2 * dim + 2)) * -1

        for idx, b in enumerate(gt_bboxes):
            # print(b, b.shape[0], gt_labels[idx])
            try:
                if b.shape[0] > 0:
                    gt_bboxes_pad[idx, :b.shape[0], :] = b
            except Exception as e:
                # print(b)
                # print(gt_labels[idx])
                # print(gt_bboxes_pad)
                # print(gt_bboxes)
                # print(gt_labels)
                raise e

        batch['gt_det'] = gt_bboxes_pad

    for k in batch_data_list[0].keys():
        if k not in ['img', 'patches_img', 'gt_det', 'img_meta']:
            try:
                gt_cls = [s[k] for s in batch_data_list]
                gt_cls = torch.stack(gt_cls, dim=0)
                batch[k] = gt_cls
            except Exception as e:
                print('Error:', k)
                [print(s['img_meta']) for s in batch_data_list]
                [print(s['img'].shape, s['img'].dtype) for s in batch_data_list]
                [print(s['gt_seg'].shape, s['img'].dtype) for s in batch_data_list]
                raise e

    return batch


def naive_collate(batch_data_list):
    # only used for inner cuda aug
    if isinstance(batch_data_list[0], list):
        batch_data_list = [j for i in batch_data_list for j in i]
        return batch_data_list
    return batch_data_list


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=False,
                     shuffle=True,
                     cuda_aug_pipeline=[],
                     persistent_workers=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): if dist
        shuffle (bool) : if shuffle
        cuda_aug_pipeline
        persistent_workers
        seed
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        CudaDataLoader: A PyTorch dataloader.
    """

    rank, world_size = get_dist_info()

    if len(cuda_aug_pipeline) == 0:  # only cpu pipeline
        cpu_collate_fn = collate
    else:
        cpu_collate_fn = naive_collate

    if not dist:
        sampler = None
        shuffle = shuffle
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    else:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        shuffle = None
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu

    if seed is not None:
        init_fn = partial(worker_init_fn_seed,
                          num_workers=num_workers, rank=rank,
                          seed=seed)
    else:
        init_fn = partial(worker_init_fn_seed,
                          num_workers=num_workers, rank=rank,
                          seed=0)

    data_loader = CudaDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=cpu_collate_fn,
        worker_init_fn=init_fn,  # to make different batch act different
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else 2,
        cuda_aug_pipeline=cuda_aug_pipeline,
        cuda_collate_fn=collate,
        **kwargs)

    return data_loader


# def worker_init_fn(worker_id):
#     passing_seed = np.random.get_state()[1][0]
#     worker_seed = passing_seed + worker_id
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#     print(f'passing_seed={passing_seed}, '
#           f'worker_id={worker_id}, '
#           f'worker_seed={worker_seed}')


def worker_init_fn_seed(worker_id, num_workers=1, rank=0, epoch=0, seed=0):
    worker_seed = worker_id + num_workers * rank + seed * (epoch + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print(f'worker_id={worker_id}, '
          f'num_workers={num_workers}, '
          f'rank={rank}, '
          f'epoch={epoch}, '
          f'seed={seed}, '
          f'worker_seed={worker_seed}')
