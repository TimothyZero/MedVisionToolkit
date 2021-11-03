from typing import Union, List
import os
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import torch

from .hook import Hook


class MonitorHook(Hook):
    def __init__(self, interval):
        self.interval = interval

    def before_iter(self, runner):
        if runner.inner_iter % self.interval == 0:
            if hasattr(runner.model, 'module'):
                runner.model.module.set_monitor_nesting(runner)
            else:
                runner.model.set_monitor_nesting(runner)

    def after_iter(self, runner):
        if runner.inner_iter % self.interval == 0:
            if hasattr(runner.model, 'module'):
                runner.model.module.reset_monitor_nesting(runner)
            else:
                runner.model.reset_monitor_nesting(runner)


class ModuleLogHook(Hook):
    def __init__(self, interval):
        self.interval = interval

    def before_iter(self, runner):
        if runner.inner_iter % self.interval == 0:
            if hasattr(runner.model, 'module'):
                runner.model.module.set_log_nesting(runner)
            else:
                runner.model.set_log_nesting(runner)

    def after_iter(self, runner):
        if runner.inner_iter % self.interval == 0:
            if hasattr(runner.model, 'module'):
                runner.model.module.reset_log_nesting(runner)
            else:
                runner.model.reset_log_nesting(runner)
