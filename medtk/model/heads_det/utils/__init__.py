#  Copyright (c) 2021. The Medical Image Computing (MIC) Lab, 陶豪毅
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

from medtk.model.heads_det.utils.util_anchors import AnchorGenerator, PointGenerator
from medtk.model.heads_det.utils.util_assigners import IoUAssigner, MaxIoUAssigner, DistAssigner, UniformAssigner
from medtk.model.heads_det.utils.util_bboxes import DeltaBBoxCoder, DistBBoxCoder
from medtk.model.heads_det.utils.util_samplers import RandomSampler, OHEMSampler, HNEMSampler
from medtk.model.heads_det.utils.util_extractors import SingleRoIExtractor, GenericRoIExtractor