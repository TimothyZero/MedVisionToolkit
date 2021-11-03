# Copyright (c) Open-MMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module
class IterTimerHook(Hook):
    def __init__(self):
        self.t = None
        self.data_time = None

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        self.data_time = time.time() - self.t
        runner.log_buffer.update({'data_time': self.data_time})

    def after_iter(self, runner):
        total_time = time.time() - self.t
        runner.log_buffer.update({'time': total_time})
        runner.log_buffer.update({'network_time': total_time - self.data_time})
        self.t = time.time()
