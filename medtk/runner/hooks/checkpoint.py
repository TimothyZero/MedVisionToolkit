# Copyright (c) Open-MMLab. All rights reserved.
# from ..dist_utils import master_only
from .hook import HOOKS, Hook


@HOOKS.register_module
class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 save_latest=True,
                 latest_optimizer=True,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.save_latest = save_latest
        self.latest_optimizer = latest_optimizer
        self.args = kwargs

    # @master_only
    def after_train_epoch(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        if self.save_latest:
            runner.save_checkpoint(
                self.out_dir, filename_tmpl='epoch_latest.pth',
                save_optimizer=self.latest_optimizer, **self.args)

        if not self.every_n_epochs(runner, self.interval):
            return

        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
