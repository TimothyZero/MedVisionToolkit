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

import logging
from collections import OrderedDict
import numpy as np

import torch.distributed as dist

logger_initialized = {}


def get_root_logger(log_file=None, log_level=logging.DEBUG, version_prefix=''):
    """
    log.info(msg) or higher will print to console and file
    log.debug(msg) will only print to file
    """

    import os
    
    logger = logging.getLogger('medtk')
    if 'medtk' in logger_initialized:
        return logger

    logger.setLevel(log_level)

    c_formatter = logging.Formatter(fmt=f'%(asctime)s - {version_prefix}|%(levelname)-6s - %(message)s',
                                    datefmt="%Y-%m-%d %H:%M:%S")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        print(f'logger {os.getpid()} rank {rank} with dist ...')
    else:
        rank = 0
        print(f'logger {os.getpid()} rank {rank} without dist ...')

    if rank == 0 and log_file:
        print('adding file handler to logger:', rank)
        f_formatter = logging.Formatter(fmt=f"%(asctime)s - {version_prefix}|%(levelname)-6s - %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S")
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

    if rank != 0:
        print('stopping logger:', rank)
        logger.setLevel(logging.WARNING)

    logger_initialized['medtk'] = True

    return logger


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True

