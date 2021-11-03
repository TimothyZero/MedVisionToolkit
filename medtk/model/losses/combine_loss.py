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

from typing import List, Union
from medtk.model.nnModules import ComponentModule


class CombineLoss(ComponentModule):
    def __init__(self,
                 loss_dicts: List[object],
                 loss_weights: list):
        super().__init__()
        assert len(loss_dicts) == len(loss_weights)
        self.weights = loss_weights
        self.losses = loss_dicts

    def __repr__(self):
        repr_str = self.__class__.__name__
        losses_str = []
        for i, l in enumerate(self.losses):
            losses_str.append(f'{self.weights[i]}x_' + l.abbr)
        repr_str += f'({",".join(losses_str)})'
        return repr_str

    def forward(self, net_output, target, weight=None):
        losses = {}
        for i, l in enumerate(self.losses):
            # print(l.abbr)
            losses[f'{self.weights[i]}x_' + l.abbr] = self.weights[i] * l(net_output, target, weight=weight)
        return losses