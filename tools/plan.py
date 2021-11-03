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

import time

from medtk.data.datasets import BasicPairDataset

# from configs.base.datasets.Test3D_det import *
# from configs.base.datasets.Test2D_det import *
# from configs.proj1.data_proj1_det import *
# from configs.base.datasets.DRIVE import *
# from configs.base.datasets.brain import *
# from configs.base.datasets.ISIC2018 import *
# from configs.base.datasets.covid_low import *
# from configs.base.datasets.covid_full import *
# from configs.base.datasets.LUNA16_multiView_det import *
from configs.datasets.HuBMAP import *

pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_seg=True),
    dict(type='Property')
]
d = BasicPairDataset(TASK, data_root + 'all_dataset.json', data_root, pipeline)
tic = time.time()
d.statistic()
print('\nTime cost:', time.time() - tic)
