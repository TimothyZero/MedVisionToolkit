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
import random
import numpy as np
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from medtk.utils import Config, get_root_logger, get_git_hash
from medtk.data import build_dataloader, MedDataParallel, MedDistributedDataParallel
from medtk.runner import build_optimizer, MedRunner


print(f"{time.strftime('%x %X')}: Process {os.getpid()} is importing/running {__file__}!")


def setup(local_rank, world_size, args):
    # Init process group
    port = min(sum(map(ord, osp.basename(args.config))) + 2000, 65535)
    print(f"Initialize Process Group {local_rank}/{world_size} at {port}... ")
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://localhost:{port}',
                            rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def init_seed(SEED, deterministic=True):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(cfg, args, logger):
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
            dist=args.dist,
            cuda_aug_pipeline=train_cuda_pipeline,
            shuffle=True,
            persistent_workers=True,
            seed=cfg.SEED
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
                min(cfg.data.workers_per_gpu, 2),
                cfg.gpus,
                dist=args.dist,
                shuffle=True,
                persistent_workers=True,
                seed=cfg.SEED
            )
        )

    return data_loaders


def worker(args):
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

    os.makedirs(osp.join(cfg.work_dir, 'logs'), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'logs', '{}.log'.format(timestamp))
    version_prefix = osp.splitext(osp.basename(cfg.filename))[0]
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, version_prefix=version_prefix)

    shutil.copy(cfg.filename, osp.join(cfg.work_dir, timestamp + "_" + osp.basename(cfg.filename)))

    logger.debug(f'Git Version: {get_git_hash()}')
    logger.debug(f'Current PID: {os.getpid()}')
    logger.debug(f'Config:\n{cfg.pretty_text}')
    logger.debug(f'Model:\n{cfg.model}')

    init_seed(cfg.SEED)

    data_loaders = build_loaders(cfg, args, logger)
    model = cfg.model
    if not args.dist:
        model = MedDataParallel(model.cuda(0), device_ids=range(cfg.gpus))  # must put model on first gpu
    else:
        model = MedDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    optimizer = build_optimizer(model, cfg.optimizer)

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


def ddp_worker(local_rank, world_size, args):
    setup(local_rank, world_size, args)

    worker(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--resume', type=str, default='latest')  # convenient resume, set 10, 40 or latest to resume
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--dist', help='distributed training', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # args.config = 'projects/liver/configs/cfg_lesion_det_rcnn_base_1.py'
    # args.config = 'projects/liver/configs/cfg_lesion_det_retina_base_1.py'
    # args.config = 'projects/liver/configs/cfg_lesion_det_retina_base_v.py'
    # args.config = 'projects/DRIVE/cfg_seg_drive.py'
    # args.config = 'projects/APCTP/configs/cfg_seg_v1.py'
    # args.config = 'projects/aneurysm/configs/cfg_baseline_v4_one_v2.py'
    # args.config = 'projects/LUNA2016/configs/cfg_baseline_one_v10_cuda.py'
    # args.config = 'projects/LUNA2016/configs/cfg_baseline_one_free_v1_cuda.py'
    # args.fold = 0
    # args.load_from = 'work_dirs/epoch_15.pth'
    # args.batch = 4
    # args.workers = 2
    # args.gpus = 2
    # args.dist = True
    # os.chdir('..')

    if args.dist:
        n_gpu = min(torch.cuda.device_count(), args.gpus)
        assert n_gpu > 1
        mp.spawn(ddp_worker, nprocs=n_gpu, args=(n_gpu, args))
    else:
        worker(args)
