import os
import torch
from torch import nn
from inspect import getframeinfo, stack
from datetime import datetime
from collections import OrderedDict
import torch.distributed as dist
from abc import abstractmethod

from medtk.model.layers.ReflectionPadNd import ReflectionPad3d
from medtk.model.nd import BatchNormNd, GroupNormNd, InstanceNormNd, LayerNormNd, ConvNd
try:
    from medtk.model.nd import DCNv1Nd, DCNv2Nd
except ImportError:
    pass


CONV_configs = {
    'Conv': dict(
        type='Conv',
        padding_mode='zeros'
    ),
    'DCNv1': dict(
        type='DCNv1',
        groups=1,
        deformable_groups=1
    ),
    'DCNv2': dict(
        type='DCNv2',
        groups=1,
        deformable_groups=1
    )
}
NORM_configs = {
    'BatchNorm': dict(
        type='BatchNorm',
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True),
    'GroupNorm': dict(
        type='GroupNorm',
        eps=1e-05,
        affine=True),
    'InstanceNorm': dict(
        type='InstanceNorm',
        eps=1e-05,
        momentum=0.1,
        affine=True,),
    'LayerNorm': dict(
        type='LayerNorm',
        eps=1e-05,
        momentum=0.1,
        affine=True
    )
}
ACT_configs = {
    'ReLU':      dict(type='ReLU', inplace=True),
    'LeakyReLU': dict(type='LeakyReLU', negative_slope=1e-2, inplace=True),
    'ELU':       dict(type='ELU', alpha=1., inplace=True),
    'PReLU':     dict(type='PReLU', num_parameters=1, init=0.25),
}


class BlockModule(nn.Module):
    def __init__(self, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()

        self.conv_cfg = CONV_configs['Conv'] if conv_cfg is None else conv_cfg
        self.norm_cfg = NORM_configs['BatchNorm'] if norm_cfg is None else norm_cfg
        self.act_cfg = ACT_configs['ReLU'] if act_cfg is None else act_cfg

    def is_conv(self, dim, obj, conv_cfg=None):
        if not conv_cfg:
            conv_cfg = self.conv_cfg.copy()
        else:
            conv_cfg = conv_cfg.copy()

        conv_type = conv_cfg['type']
        if conv_type == 'Conv':
            conv_layer = ConvNd(dim)
        elif conv_type == 'DCNv1':
            conv_layer = DCNv1Nd(dim)
        else:
            raise NotImplementedError
        return isinstance(obj, conv_layer)

    def is_norm(self, dim, obj, norm_cfg=None):
        if not norm_cfg:
            norm_cfg = self.norm_cfg.copy()
        else:
            norm_cfg = norm_cfg.copy()

        norm_type = norm_cfg['type']
        if norm_type == 'BatchNorm':
            norm_layer = BatchNormNd(dim)
        elif norm_type == 'GroupNorm':
            norm_layer = GroupNormNd(dim)
        elif norm_type == 'InstanceNorm':
            norm_layer = InstanceNormNd(dim)
        elif norm_type == 'LayerNorm':
            norm_layer = LayerNormNd(dim)
        else:
            raise NotImplementedError
        return isinstance(obj, norm_layer)

    def build_conv(self, dim, in_channels, out_channels, conv_cfg=None, **kwargs):
        if not conv_cfg:
            conv_cfg = self.conv_cfg.copy()
        else:
            conv_cfg = conv_cfg.copy()

        conv_type = conv_cfg.pop('type')
        for k, v in CONV_configs[conv_type].items():
            if k not in conv_cfg.keys() and k != 'type':
                conv_cfg[k] = v
        for k, v in kwargs.items():
            conv_cfg[k] = v

        if conv_type == 'Conv':
            if conv_cfg.get('padding_mode') == 'reflect' and dim == 3:
                padding = conv_cfg.pop('padding') if 'padding' in conv_cfg.keys() else 0
                padding_mode = conv_cfg.pop('padding_mode')
                return nn.Sequential(
                    ReflectionPad3d(padding=padding),
                    ConvNd(dim)(in_channels, out_channels, **conv_cfg)
                )
            else:
                return ConvNd(dim)(in_channels, out_channels, **conv_cfg)
        elif conv_type == 'DCNv1':
            if conv_cfg.get('padding_mode') == 'reflect' and dim == 3:
                padding = conv_cfg.pop('padding') if 'padding' in conv_cfg.keys() else 0
                padding_mode = conv_cfg.pop('padding_mode')
                return nn.Sequential(
                    ReflectionPad3d(padding=padding),
                    DCNv1Nd(dim)(in_channels, out_channels, **conv_cfg)
                )
            else:
                return DCNv1Nd(dim)(in_channels, out_channels, **conv_cfg)
        else:
            raise NotImplementedError

    def build_norm(self, dim, num_channels, norm_cfg=None, **kwargs):
        if not norm_cfg:
            norm_cfg = self.norm_cfg.copy()
        else:
            norm_cfg = norm_cfg.copy()

        norm_type = norm_cfg.pop('type')
        for k, v in NORM_configs[norm_type].items():
            if k not in norm_cfg.keys() and k != 'type':
                norm_cfg[k] = v
        for k, v in kwargs.items():
            norm_cfg[k] = v

        if norm_type == 'BatchNorm':
            return BatchNormNd(dim)(num_channels, **norm_cfg)
        elif norm_type == 'GroupNorm':
            num_groups = norm_cfg.pop('num_groups')
            return GroupNormNd(dim)(num_groups, num_channels, **norm_cfg)
        elif norm_type == 'InstanceNorm':
            return InstanceNormNd(dim)(num_channels, **norm_cfg)
        elif norm_type == 'LayerNorm':
            return LayerNormNd(dim)(num_channels, **norm_cfg)
        else:
            raise NotImplementedError

    def build_act(self, act_cfg=None, **kwargs):
        if not act_cfg:
            act_cfg = self.act_cfg.copy()
        else:
            act_cfg = act_cfg.copy()

        act_type = act_cfg.pop('type')
        for k, v in ACT_configs[act_type].items():
            if k not in act_cfg.keys() and k != 'type':
                act_cfg[k] = v
        for k, v in kwargs.items():
            act_cfg[k] = v

        if act_type == 'ReLU':
            return nn.ReLU(**act_cfg)
        elif act_type == 'LeakyReLU':
            return nn.LeakyReLU(**act_cfg)
        elif act_type == 'ELU':
            return nn.ELU(**act_cfg)
        elif act_type == 'PReLU':
            return nn.PReLU(**act_cfg)
        else:
            raise NotImplementedError

    def get(self, k, default=None):
        try:
            return self.__getattribute__(k)
        except:
            return default


class ComponentModule(BlockModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_enable = False
        self.monitor_enable = False

        self.runner_data_meta = None
        self.runner_data = None

    def try_to_info(self, *args):
        if self.log_enable:
            caller = getframeinfo(stack()[1][0])
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                  "- {} - line {} - {} :".format(self.__class__.__name__, caller.lineno, caller.function), end='')
            self.print(*args)

    def info(self, *args):
        caller = getframeinfo(stack()[1][0])
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
              "- {} - line {} - {} :".format(self.__class__.__name__, caller.lineno, caller.function), end='')
        self.print(*args)

    @staticmethod
    def print(*args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 1:
                print('\n\t', end='')
            print(arg, end=' ')
        print('')

    def set_log(self, runner):
        self.runner_data = runner.data_batch
        self.runner_data_meta = {
            'result_dir': os.path.join(runner.work_dir, f"{runner.mode}_results"),
            'mode': runner.mode,
        }
        self.log_enable = True

    def reset_log(self, runner):
        self.runner_data = None
        self.runner_data_meta = None
        self.log_enable = False

    def set_monitor(self, runner):
        self.runner_data = runner.data_batch
        self.runner_data_meta = {
            'result_dir': os.path.join(runner.work_dir, f"{runner.mode}_results"),
            'mode': runner.mode,
        }
        self.monitor_enable = True

    def reset_monitor(self, runner):
        self.runner_data = None
        self.runner_data_meta = None
        self.monitor_enable = False

    def print_model_params(self):
        print(self.__class__.__name__)
        t = 0
        for k, p in self.named_parameters():
            t += p.numel()
            print('{}\t{}\t{}'.format(k, p.numel(), t))

        total_params = sum(p.numel() for p in self.parameters())
        print('Total params：{}'.format(total_params))

        total_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total trainable parameters：{}'.format(total_trainable_parameters))
        print('\n')


class BaseTask(ComponentModule):
    def __init__(self):
        super().__init__()

    def set_log_nesting(self, runner):
        for child in self.children():
            if hasattr(child, 'set_log'):
                child.set_log(runner)

    def reset_log_nesting(self, runner):
        for child in self.children():
            if hasattr(child, 'reset_log'):
                child.reset_log(runner)

    def set_monitor_nesting(self, runner):
        for child in self.children():
            if hasattr(child, 'set_monitor'):
                child.set_monitor(runner)

    def reset_monitor_nesting(self, runner):
        for child in self.children():
            if hasattr(child, 'reset_monitor'):
                child.reset_monitor(runner)

    @staticmethod
    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            # print(loss_name, loss_value)
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.float().mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.float().mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        # print(loss)
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @abstractmethod
    def forward_train(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img, gt_seg ...
            *args:
            **kwargs:

        Returns:
            losses: dict
            prediction: None
            net_output: tensor, monitored features
        """
        pass

    @abstractmethod
    def forward_valid(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img, gt_seg ...
            *args:
            **kwargs:

        Returns:
            losses + metric: dict
            prediction: prediction with probability which is used in metric and inference
            net_output: tensor, monitored features
        """
        pass

    @abstractmethod
    def forward_infer(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img
            *args:
            **kwargs:

        Returns:
            prediction: prediction with probability which is used in metric and inference
            net_output: tensor, monitored features
        """
        pass

    def forward(self, data_batch, return_loss=True, *args, **kwargs):
        assert all([i in data_batch.keys() for i in ['img_meta', 'img']]), print(data_batch.keys())
        assert 'gt' in str(data_batch.keys()), 'Only support train and valid! Using forward_infer for inference!'
        assert torch.max(data_batch['img']) <= 10.0 and torch.min(data_batch['img']) >= -10.0, \
            'max={} min={}'.format(torch.max(data_batch['img']).item(), torch.min(data_batch['img']).item())

        if not return_loss:
            out = self.forward_valid(data_batch, *args, **kwargs)
        else:
            out = self.forward_train(data_batch, *args, **kwargs)

        assert len(out) == 3, 'forward function must return three outputs'
        assert isinstance(out[0], dict)

        losses_dict, prediction, net_output = out
        # print('after forward', losses_dict)
        loss, log_vars = self.parse_losses(losses_dict)
        log_outputs = dict(loss=loss,
                           log_vars=log_vars,
                           num_samples=data_batch['img'].shape[0])
        # print('after parse', log_outputs)
        return log_outputs, prediction, net_output

    # def train_iter(self, data_batch, optimizers):
    #     losses_dict, prediction, net_output = self.forward_train(data_batch)
    #     loss, log_vars = self._parse_losses(losses_dict)
    #     outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch['img'].data))
    #     return outputs, prediction, net_output
    #
    # def valid_iter(self, data_batch, optimizers):
    #     # print("valid...")
    #     metrics_losses, prediction, net_output = self.forward_valid(data_batch)
    #     loss, log_vars = self._parse_losses(metrics_losses)
    #     outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch['img'].data))
    #     return outputs, prediction, net_output


if __name__ == '__main__':

    block = BlockModule()
    c = block.build_conv(dim=3, in_channels=1, out_channels=3, kernel_size=3)
    print(c)
    c = block.build_conv(dim=3, in_channels=1, out_channels=3, kernel_size=3, conv_cfg=CONV_configs['DCNv1'].copy())
    print(c)
    n = block.build_norm(dim=3, num_channels=12, norm_cfg=NORM_configs['GroupNorm'].copy(), num_groups=4)
    print(n)
    n = block.build_norm(dim=3, num_channels=12, norm_cfg=NORM_configs['GroupNorm'].copy(), num_groups=2)
    print(n)
    n = block.build_norm(dim=3, num_channels=12, norm_cfg=NORM_configs['InstanceNorm'].copy())
    print(n)