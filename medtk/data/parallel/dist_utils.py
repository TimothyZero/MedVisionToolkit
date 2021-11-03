# Copyright (c) Open-MMLab. All rights reserved.
import functools
import os
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TORCH_VERSION = torch.__version__


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ["SLURM_NPROCS"])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    print("dist-url {}:{} at PROCID {} / {}".format(addr, port, proc_id, world_size))
    print({
        'proc_id': proc_id,
        'world_size': world_size,
        'ntasks': ntasks,
        'num_gpus': num_gpus,
        'addr': addr,
        'port': port,
    })
    dist.init_process_group(backend=backend)


def get_dist_info():
    if TORCH_VERSION < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
#
#
# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2020/4/29 下午12:00
# # @Author  : huangyechong
# # @Site    :
# # @File    : dist_utils.py
# # @Software: PyCharm
#
#
# '''
# @Author: Jiamin Ren
# @Date: 2020-04-29 09:56:57
# '''
# import os
# import time
# import socket
#
# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
#
# __all__ = [
#     'init_dist', 'init_dist_slurm', 'broadcast_params', 'average_gradients']
#
#
# # 在这里不行, 一个报了runtime error之后, 其他的进程上创建的进程还是成功的
# def retry_decorator(max_retry=5, wait=3, exception=RuntimeError, default_output=None):
#     """
#     用来重试可能失败的函数
#     :param max_retry: 最大的重试次数
#     :param wait: 重试的等待时间
#     :param exception: 只有遇到这些错误才会重试, 允许单个错误, e.g. RuntimeError; 或元组, e.g. (RuntimeError, AttributeError)
#     :param default_output: 重试失败时默认的返回
#     :return:
#     """
#     from time import sleep
#     from functools import wraps
#
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             output = default_output
#             for i in range(1, max_retry + 1):
#                 try:
#                     output = func(*args, **kwargs)
#                     break
#                 except exception as e:
#                     print('Failed at attempt no.{}, got expected error: \n{}\nRetry in {} seconds'.format(
#                         i, e, wait  # type(e).__name__
#                     ))
#                     sleep(wait)
#                     continue
#                 except Exception as e:
#                     print('Failed at attempt no.{}, got unexpected error: \n{}\nUsing default output {}'.format(
#                         i, e, default_output  # type(e).__name__
#                     ))
#                     break
#             else:
#                 print('Retry exceeds the max retry count {}, using default output {}'.format(max_retry, default_output))
#             return output
#
#         return wrapper
#
#     return decorator
#
#
# def init_dist(backend='nccl',
#               master_ip='127.0.0.1',
#               port=None):
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method('fork')
#
#     os.environ['MASTER_ADDR'] = master_ip
#
#     assert port is not None
#
#     os.environ['MASTER_PORT'] = str(port)
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(rank % num_gpus)
#     dist.init_process_group(backend=backend)
#     return rank, world_size
#
#
# def init_dist_slurm(backend='nccl', port=None):
#     # if mp.get_start_method(allow_none=True) != 'fork':
#     #     mp.set_start_method('fork')
#     proc_id = int(os.environ['SLURM_PROCID'])
#     ntasks = int(os.environ['SLURM_NTASKS'])
#     node_list = os.environ['SLURM_NODELIST']
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(proc_id % num_gpus)
#
#     if '[' in node_list:
#         beg = node_list.find('[')
#         pos1 = node_list.find('-', beg)
#         if pos1 < 0:
#             pos1 = 1000
#         pos2 = node_list.find(',', beg)
#         if pos2 < 0:
#             pos2 = 1000
#         node_list = node_list[:min(pos1, pos2)].replace('[', '')
#     addr = node_list[8:].replace('-', '.')
#
#     os.environ['MASTER_ADDR'] = addr
#
#     assert port is not None
#
#     os.environ['MASTER_PORT'] = str(port)
#     os.environ['WORLD_SIZE'] = str(ntasks)
#     os.environ['RANK'] = str(proc_id)
#     dist.init_process_group(backend='nccl')
#
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     return rank, world_size
#
#
# def average_gradients(model):
#     for param in model.parameters():
#         if param.requires_grad and not (param.grad is None):
#             dist.all_reduce(param.grad.data)
#
#
# def broadcast_params(model):
#     for p in model.state_dict().values():
#         dist.broadcast(p, 0)
