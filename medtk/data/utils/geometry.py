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

import numpy as np


def getSphere(dim, size, diameter):
    radius = diameter / 2 - 0.5
    structure = np.zeros((size,) * dim)

    center = [i / 2 for i in structure.shape]
    ctr = np.meshgrid(*[np.arange(0.5, size)]*dim, indexing='ij')
    ctr = np.stack(ctr, axis=0)
    ctr = np.transpose(ctr, [*range(1, dim + 1), 0])

    distance = np.sum(np.power(np.abs(ctr - center), 2), axis=-1)
    structure = (distance <= radius ** 2).astype(np.float32)
    return structure