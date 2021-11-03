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

from .dice_loss import SoftDiceLoss, DiceLoss, NNUnetSoftDiceLoss  # DiceAndCrossEntropyLoss,
from .cross_entropy_loss import CrossEntropyLoss
# from .focal_loss import FocalLoss
from .smooth_l1_loss import SmoothL1Loss
from .fcloss import SigmoidFocalLoss
# from .mm_focal_loss import MMFocalLoss
from .combine_loss import CombineLoss
from .l1loss import BalancedL1Loss
from .iou_loss import IoULoss, GIoULoss, DIoULoss
