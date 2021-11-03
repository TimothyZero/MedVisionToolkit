import numpy as np
import time
from datetime import datetime
import random
import argparse
import os
import os.path as osp
import torch
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Dataset, DataLoader

from medtk.data.parallel.dist_utils import get_dist_info
from medtk.data.loader.cudadataloader import CudaDataLoader
from medtk.data.loader.sampler import DistributedSampler
from medtk.utils import get_root_logger
PRINT_FUN = lambda x: print(f"{datetime.now().strftime('%x %X.%f')}[{os.getpid()}]:", x)  # PRINT_FUN


PRINT_FUN(f"I'm importing/running {__file__}!")

# logger = get_root_logger('./t2.txt')


def init_seed(SEED, deterministic=True):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn_seed(worker_id, num_workers=1, rank=0, epoch=0, seed=0):
    worker_seed = worker_id + num_workers * rank + seed * (epoch + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    PRINT_FUN(f'pid={os.getpid()}/{os.getppid()}, '
              f'worker_id={worker_id}, '
              f'num_workers={num_workers}, '
              f'rank={rank}, '
              f'epoch={epoch}, '
              f'seed={seed}, '
              f'worker_seed={worker_seed}')


class Toy(Dataset):
    def __init__(self, transforms):
        self.transforms = transforms

    def __len__(self):
        return 20

    def __getitem__(self, item):
        # PRINT_FUN(f'getting {item}')
        x = item

        for t in self.transforms:
            x = t(x)
        return x


class RandomAdd(object):
    def __call__(self, x: int):
        if isinstance(x, int):
            return x + random.uniform(0, 1)
        if isinstance(x, float):
            return x + random.uniform(0, 1)
        if isinstance(x, list):
            return [i + random.uniform(0, 1) for i in x]


class RandomScale(object):
    def __call__(self, x: int):
        return x * random.random()


def setup(local_rank, world_size, args):
    # Init process group
    port = min(sum(map(ord, osp.basename(args.config))) + 2000, 65535)
    PRINT_FUN(f"Initialize Process Group {local_rank}/{world_size} at {port}... ")
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://localhost:{port}',
                            rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def worker(args):
    import matplotlib.pyplot as plt
    # import multiprocessing
    #
    # multiprocessing.set_start_method('spawn')  # ensure seed passing work!

    # print('sys state',np.random.get_state()[1][:10])

    seed = 457

    # model init
    init_seed(seed)
    model = torch.nn.Conv2d(1, 1, 1)
    PRINT_FUN(model.weight.sum())

    dataset = Toy(
        transforms=[
            RandomAdd(),
        ]
    )

    batch_size = 4
    num_workers = 2
    shuffle = True
    persistent_workers = False

    rank, world_size = get_dist_info()

    init_fn = partial(
        worker_init_fn_seed,
        num_workers=num_workers,
        rank=rank,
        seed=seed)

    if args.dist:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        shuffle = None
    else:
        sampler = None
        shuffle = shuffle

    data_loader = CudaDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=shuffle,  # if worker_init is always the same across epochs, this will change the data order
        persistent_workers=persistent_workers,  # ignore 'epoch seed'，saving time with prefetcher,
        worker_init_fn=init_fn
    )

    all_data = []
    for epoch in range(3):
        try:
            PRINT_FUN(f'data inner epoch {data_loader.epoch + 1}')
        except Exception as e:
            print(e)
        PRINT_FUN(f'epoch {epoch + 1} start')
        for i, batch in enumerate(data_loader):
            # print(data_loader.epoch)
            PRINT_FUN(f'batch{i} = {batch}')
            PRINT_FUN(f'batch random {np.random.rand()}')
            # all_data += batch.cpu().numpy().tolist()

        PRINT_FUN(f'epoch {epoch + 1} end\n')
    PRINT_FUN(sorted(all_data))


def ddp_worker(local_rank, world_size, args):
    setup(local_rank, world_size, args)

    worker(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgggg')
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

    if args.dist:
        n_gpu = min(torch.cuda.device_count(), args.gpus)
        assert n_gpu > 1
        mp.spawn(ddp_worker, nprocs=n_gpu, args=(n_gpu, args))
    else:
        worker(args)
