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
import time
import numpy as np
import torch
import argparse

from medtk.utils import Config
from medtk.data import build_dataloader
from medtk.data.pipelines import Viewer, Display


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--dataset', type=str, default='train')
args = parser.parse_args()

if not args.config:
    quit(-1)

cfg = Config.fromfile(args.config)

v = Viewer()
d = Display()

dataset = cfg.data[args.dataset]
print(dataset.pipeline)
print("dataset length", len(dataset))

for i in range(len(dataset)):
    dataset.setLatitude(min(1 + 0.1 * i, 1))
    results = dataset.__getitem__(i)
    print('No.', i, "Latitude", dataset.getLatitude(), len(results))
    # a = train_cuda_aug_pipeline[0]
    # t_results = a(results)
    result = results[0]
    print(result.get('filename', result.get('img_meta', [])['filename']))
    print(result['img'].shape)
    if 'gt_det' in result.keys():
        print(result['gt_det'])
    d(result)
    v(result)
    if i == 10:
        break

import random

SEED = 0
# for running reproduction
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


loader = build_dataloader(dataset,
                          imgs_per_gpu=1,
                          workers_per_gpu=2,
                          shuffle=False,
                          # sampler=SubsetRandomSampler(range(12))
                          )

tic = time.time()
for epoch in range(1):
    # loader.dataset.setLatitude(0.1 * epoch)
    np.random.seed(SEED + epoch)
    print('epoch ', epoch, np.random.get_state()[1][0])

    for i, batch in enumerate(loader):
        print(batch['img'].shape)
        print('\n', f"epoch{epoch}-iter{i}-innerP = {getattr(loader.dataset, 'latitude', None)}")
        print(batch['img_meta'][0]['time'])
        # for img_meta in batch['img_meta']:
        #     print(img_meta)

        # for k, v in batch.items():
        #     print(k)
        #     if not isinstance(v, (tuple, list, dict)):
        #         print(v.shape)
        if i == 10:
            break
print(time.time() - tic)