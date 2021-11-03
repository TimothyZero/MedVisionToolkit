import torch
from torch import nn
import numpy as np

from medtk.model.nnModules import ComponentModule


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be )
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:(b, c, d, h, w)
    :param gt: (b, c, d, h, w) | (b, 1, d, h, w) | (b, d, h, w)
    :param axes:
    :param mask: (b, 1, d, h, w), mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    # print(tp, fp, fn)

    return tp, fp, fn


class NNUnetSoftDiceLoss(ComponentModule):
    def __init__(self):
        """
        [Soft] Directly use the predicted probabilities instead of doing threshold and
        converting them into a binary mask.
        """
        super(NNUnetSoftDiceLoss, self).__init__()
        self.abbr = "nn_sf_dice_loss"
        self.square = False
        self.do_bg = False
        self.apply_nonlin = nn.Softmax(dim=1)
        self.smooth = 1e-5

    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                weight=None):
        """

        Args:
            net_output:
            target:
            weight

        Returns:

        """
        shp_x = net_output.shape

        # average on batch
        axes = [0] + list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        tp, fp, fn = get_tp_fp_fn(net_output, target, axes, weight, self.square)

        # if batch dice: (c,), else (b,c)
        dc = torch.true_divide(2 * tp + self.smooth, 2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            dc = dc[1:]
        dc = dc.mean()
        return 1 - dc


class SoftDiceLoss(ComponentModule):
    def __init__(self, square=False, do_bg=False):
        """
        [Soft] Directly use the predicted probabilities instead of doing threshold and
        converting them into a binary mask.
        """
        super(SoftDiceLoss, self).__init__()
        self.abbr = "sf_dice_loss"
        self.square = square
        self.do_bg = do_bg
        self.apply_nonlin = nn.Softmax(dim=1)
        self.smooth = 1
    
    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                weight=None):
        """

        Args:
            net_output:
            target:

        Returns:

        """

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        with torch.no_grad():
            if net_output.shape == target.shape:
                target_onehot = target
            else:
                target = target.long()
                target_onehot = torch.zeros_like(net_output)
                target_onehot.scatter_(1, target.unsqueeze(1), 1)  # torch 1.7.0
        assert net_output.shape == target_onehot.shape

        if not self.do_bg:
            target_onehot = target_onehot[:, 1:]
            net_output = net_output[:, 1:]

        axes = [0] + list(range(2, len(target_onehot.shape)))
        intersection = torch.sum(target_onehot * net_output, dim=axes)
        union = torch.sum(net_output, dim=axes) + torch.sum(target_onehot, dim=axes)
        dice = torch.true_divide(2. * intersection + self.smooth, union + self.smooth)
        assert len(dice) == target_onehot.shape[1]
        dice = dice.mean()
        return 1 - dice


class DiceLoss(ComponentModule):
    def __init__(self):
        super().__init__()
        self.abbr = "dice_loss"
        self.square = False
        self.do_bg = False
        self.apply_nonlin = nn.Softmax(dim=1)
        self.smooth = 1e-5

    def forward(self,
                net_output: torch.Tensor,
                target: torch.Tensor,
                weight=None):
        """

        Args:
            net_output:
            target:
            weight:

        Returns:

        """
        assert net_output.ndim == target.ndim, f'Dimension not matched! {net_output.shape}, {target.shape}'

        with torch.no_grad():
            target = target.long()
            target_onehot = torch.zeros_like(net_output.shape)
            target_onehot.scatter_(1, target, 1)

            prediction = torch.argmax(net_output, dim=1, keepdim=True)
            prediction_onehot = torch.zeros_like(net_output)
            prediction_onehot.scatter_(1, prediction, 1)

        assert prediction_onehot.shape == target_onehot.shape

        if not self.do_bg:
            target_onehot = target_onehot[:, 1:]
            prediction_onehot = prediction_onehot[:, 1:]

        intersection = torch.sum(target_onehot * prediction_onehot)
        union = torch.sum(prediction_onehot) + torch.sum(target_onehot)
        dice = torch.true_divide(2. * intersection + self.smooth, union + self.smooth)
        return 1 - dice


# class DiceAndCrossEntropyLoss(NetModule):
#     def __init__(self, cross_section=False, reduction="mean", weight=None):
#         super(DiceAndCrossEntropyLoss, self).__init__()
#         self.abbr = 'dc&ce_loss'
#         self.reduction = reduction
#         self.cross_section = cross_section
#
#         if not cross_section:
#             if weight is None:
#                 weight = [1.0, 2.0]
#             self.weight = weight
#             assert len(self.weight) == 2
#         else:
#             if weight is None:
#                 weight = [1.0, 1.0, 1.0, 1.0]
#             self.weight = weight
#             assert len(self.weight) == 4
#
#         self.ce = CrossEntropyLoss(reduction='mean')
#         self.dc = SoftDiceLoss()
#
#     def forward(self, net_output, target):
#         '''
#         net_output: torch.Size([2, class + 1, 128, 128, 128])
#         target    : torch.Size([2, [1,] 128, 128, 128]) 0-class
#         '''
#         # print(net_output.shape)
#         # print(target.shape)
#         # print(torch.sum(target == 1), torch.sum(target == 2), torch.sum(target == 3), torch.sum(target == 4))
#
#         dc_loss = - self.dc(net_output, target)
#         ce_loss = self.ce(net_output, target)
#
#         n_classes = net_output.shape[1] - 1
#         if self.cross_section and n_classes > 1:
#             dc_loss_intersections = torch.zeros(n_classes).to(device=net_output.device)
#             for cls in range(1, n_classes + 1):
#                 target_intersection = ((target > 0) & (target != cls)).type_as(target)
#                 output_intersection = net_output[:, [0, cls], ...]
#                 dc_loss_intersections[cls - 1] = self.dc(output_intersection, target_intersection)
#
#             dc_loss_inter = torch.mean(dc_loss_intersections)
#
#             target_union = (target >= 1).type_as(target)
#             output_union = torch.cat((net_output[:, :1, ...],
#                                       torch.max(net_output[:, 1:, ...], dim=1, keepdim=True)[0]), dim=1)
#             dc_loss_union = - self.dc(output_union, target_union)
#
#             # print(ce_loss, dc_loss, dc_loss_union, dc_loss_intersection)
#             result = {f"{self.weight[0]}x_ce_loss": self.weight[0] * ce_loss,
#                       f"{self.weight[1]}x_dc_loss": self.weight[1] * dc_loss,
#                       f"{self.weight[2]}x_dc_union_loss": self.weight[2] * dc_loss_union,
#                       f"{self.weight[3]}x_dc_inter_loss": self.weight[3] * dc_loss_inter}
#         else:
#             # print(ce_loss, dc_loss)
#             result = {f"{self.weight[0]}x_ce_loss": self.weight[0] * ce_loss,
#                       f"{self.weight[1]}x_dc_loss": self.weight[1] * dc_loss}
#         return result
