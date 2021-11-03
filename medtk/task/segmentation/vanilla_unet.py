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

from .simple_segment import SimpleSegment


class VanillaUNet(SimpleSegment):
    def __init__(self, DIM, IN, BASE, LEVELS, CLASSES, loss_cls=None, metrics=None):
        if loss_cls is None:
            loss_cls = dict(type='CrossEntropyLoss')
        if metrics is None:
            metrics = [
                dict(type='Dice', aggregate=None),
                dict(type='AUC', aggregate=None),
                dict(type='IOU', task='SEG', aggregate=None)
            ]
        backbone = dict(
            type='UNetEncoder',
            dim=DIM,
            in_channels=IN,
            base_width=BASE,
            out_indices=list(range(LEVELS))
        )
        neck = dict(
            type='UNetDecoder',
            dim=DIM,
            in_channels=[BASE * 2 ** i for i in range(LEVELS)],
            out_indices=[0]
        )
        head = dict(
            type='BaseSegHead',
            dim=DIM,
            in_channels=[BASE],
            num_classes=CLASSES,
            loss_cls=loss_cls,
            metrics=metrics
        )
        super(VanillaUNet, self).__init__(DIM, backbone, neck, head)
