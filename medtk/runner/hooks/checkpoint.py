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
                 save_best=True,
                 latest_optimizer=True,
                 start_from=0,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.save_latest = save_latest
        self.save_best = save_best
        self.latest_optimizer = latest_optimizer
        self.start_from = start_from
        self.args = kwargs
        self.pre_best = 0
        self.iter_best_metric = []

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
        if runner.epoch + 1 >= self.start_from:
            runner.save_checkpoint(
                self.out_dir, save_optimizer=self.save_optimizer, **self.args)

    def after_val_iter(self, runner):
        self.iter_best_metric.append(runner.outputs['log_vars'].get('best', -1))

    def after_val_epoch(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir
        if self.save_best:
            cur_best = sum(self.iter_best_metric) / len(self.iter_best_metric)
            runner.logger.info(f'Best metric: {self.pre_best} => {cur_best}.')
            if self.pre_best < cur_best:
                runner.logger.info(f'Saving best at epoch {runner.epoch}.')
                self.pre_best = cur_best
                runner.save_checkpoint(
                    self.out_dir, filename_tmpl='epoch_best.pth',
                    save_optimizer=self.latest_optimizer, **self.args)

        self.iter_best_metric = []