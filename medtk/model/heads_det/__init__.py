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

from medtk.model.heads_det.dense_heads.deeplung_head import DeepLungHead
from medtk.model.heads_det.dense_heads.retina_head import RetinaHead
from medtk.model.heads_det.dense_heads.fcos_head import FCOSHead
from medtk.model.heads_det.dense_heads.rpn_head import RPNHead

from medtk.model.heads_det.roi_heads.bbox_heads import *
from medtk.model.heads_det.roi_heads.roi_head import ROIHead

from medtk.model.heads_det.utils import *