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
from torch import nn

from medtk.model.nnModules import ComponentModule
from medtk.model.nd import ConvNd, BatchNormNd, DropoutNd


def discriminator_block(dim, in_filters, out_filters, bn=True):
    block = [
        ConvNd(dim)(in_filters, out_filters, 3, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        DropoutNd(dim)(0.25)
    ]
    if bn:
        block.append(BatchNormNd(dim)(out_filters, 0.8))
    return block


class Discriminator(ComponentModule):
    def __init__(self, dim, img_size, channels=1):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.model = nn.Sequential(
            *discriminator_block(self.dim, channels, 16, bn=False),
            *discriminator_block(self.dim, 16, 32),
            *discriminator_block(self.dim, 32, 64),
            *discriminator_block(self.dim, 64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 3, 1), nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvNd(self.dim)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, BatchNormNd(self.dim)):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
