# -*- coding: utf-8 -*
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
import os
import os.path as osp
import shutil
import torch
import time
from datetime import datetime
import random
import numpy as np
import argparse
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.multiprocessing import set_start_method

import torch.distributed as dist
import torch.multiprocessing as mp

from medtk.utils import Config, DictAction, get_root_logger, get_git_hash
from medtk.data import build_dataloader, MedDataParallel, MedDistributedDataParallel, init_dist, get_dist_info
from medtk.runner import build_optimizer, MedRunner


print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Process {os.getpid()} is importing/running {__file__}!")


def init_seed(SEED, deterministic=True):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_task(cfg):
    assert cfg.TASK in ('SEG', 'CLS', 'DET')
    if isinstance(cfg.model, dict):
        cfg.model['dim'] = cfg.DIM
    if cfg.TASK.upper() == 'SEG':
        model = cfg.model
    elif cfg.TASK.upper() == 'CLS':
        model = cfg.model
    elif cfg.TASK.upper() == 'DET':
        model = cfg.model
    else:
        raise NotImplementedError
    return model


def build_loaders(cfg, args, distributed=False):
    train_cuda_pipeline = cfg.get('train_cuda_pipeline', [])
    if hasattr(cfg.data, 'all'):
        assert NotImplementedError

    data_loaders = []
    datasets = [cfg.data.train]
    if args.fold >= 0:
        datasets[-1].set_exclude_fold(args.fold)
    logger.info(f'Train {len(datasets[0])}')
    data_loaders.append(
        build_dataloader(
            datasets[-1],
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=distributed,
            cuda_aug_pipeline=train_cuda_pipeline,
            shuffle=True
        )
    )

    if len(cfg.workflow) == 2:
        datasets.append(cfg.data.valid)
        if args.fold >= 0:
            datasets[-1].set_include_fold(args.fold)
        logger.info(f'Valid {len(datasets[1])}')
        data_loaders.append(
            build_dataloader(
                datasets[-1],
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                cfg.gpus,
                dist=distributed,
                shuffle=True
            )
        )

    return data_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=457)
    parser.add_argument('--resume', type=str, default='latest')  # convenient resume, set 10, 40 or latest to resume
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    args = parse_args()

    """ Usage
    CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 1 tools/train_dist.py --launcher pytorch --config projects/LUNA2016/configs/cfg_v1_two_v1.py --fold 9 --batch 28 --workers 12 --gpus 2
    """

    # args.config = 'projects/liver/configs/cfg_lesion_det_rcnn_base_1.py'
    # args.batch = 1
    # args.workers = 1

    cfg = Config.fromfile(args.config)

    if args.batch:
        cfg.data.imgs_per_gpu = args.batch
    if args.workers > -1:
        cfg.data.workers_per_gpu = args.workers
    if args.gpus:
        cfg.gpus = args.gpus
    if args.lr:
        cfg.optimizer.lr = args.lr

    if args.fold < 0:
        cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name)
    else:
        cfg.work_dir = osp.join(cfg.work_dir, cfg.module_name, str(args.fold))

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank = int(os.environ['LOCAL_RANK'])
        num_gpus = torch.cuda.device_count()
        _, world_size = get_dist_info()

    os.makedirs(osp.join(cfg.work_dir, 'logs'), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'logs', '{}.log'.format(timestamp))
    version_prefix = osp.splitext(osp.basename(cfg.filename))[0]
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, version_prefix=version_prefix)

    shutil.copy(cfg.filename, osp.join(cfg.work_dir, timestamp + "_" + osp.basename(cfg.filename)))

    init_seed(args.seed)
    data_loaders = build_loaders(cfg, args, distributed)
    model = build_task(cfg)
    if distributed:
        model = MedDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    else:
        model = MedDataParallel(model.cuda(0), device_ids=range(cfg.gpus))  # must put model on first gpu
    optimizer = build_optimizer(model, cfg.optimizer)

    logger.debug(f'Version:\n{get_git_hash()}\n\n')
    logger.debug(f'Current PID: {os.getpid()}')
    logger.debug(f'init seed {args.seed}')
    logger.debug(f'Config:\n{cfg.pretty_text}')
    logger.debug(f'Model:\n{model.module.__repr__()}')

    runner = MedRunner(
        model,
        optimizer,
        cfg.work_dir,
        logger=logger,
        timestamp=timestamp
    )

    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   getattr(cfg, 'augmentation_cfg', None))
    if args.resume:
        if osp.exists(cfg.work_dir + f'/epoch_{args.resume}.pth'):
            runner.resume(cfg.work_dir + f'/epoch_{args.resume}.pth')
        else:
            logger.info(f"Checkpoint {cfg.work_dir + f'/epoch_{args.resume}.pth'} not exists!")
    else:
        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        elif args.load_from:
            runner.load_checkpoint(args.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
