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

import torch
from medtk.model.nnModules import ComponentModule


class Acc(ComponentModule):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, pred, gt):
        """

        :param gt: (b)
        :param pred: (b, classes)
        :return:
        """
        acc = (gt == pred.argmax(-1)).float().detach().numpy()
        acc = float(100 * acc.sum() / len(acc))
        print(acc, pred.argmax(-1), gt)
        return {"Acc": acc}