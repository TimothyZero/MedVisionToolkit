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

from .unet_dec import UNetDecoder
from .resunet_dec import ResUNetDecoder
from .vnet_dec import VBNetDecoder
from .iternet import IterUNetDecoder
from .fpn import FPN
from .dec import Decoder

from .convlstm_dec import ConvLSTMDecoder
from .convgru_dec import ConvGRUDecoder

from .dcgan_dis import Discriminator
