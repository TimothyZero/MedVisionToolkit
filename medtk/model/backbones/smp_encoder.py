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

# -*- coding:utf-8 -*-
from typing import Optional, Union, List
from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3Decoder

from ..nnModules import ComponentModule


class SmpEncoder(ComponentModule):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 in_channels: int = 3,
                 encoder_depth: int = 5,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,):
        super(SmpEncoder, self).__init__()
        self.dim = 2
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.encoder(x)
        # for i in features:
        #     print(i.shape)
        decoder_output = self.decoder(*features)
        # for o in decoder_output:
        #     print(o.shape)
        # print(decoder_output.shape)
        return [decoder_output]
