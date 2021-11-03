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
import torch.nn.functional as F
from collections import OrderedDict

from medtk.model.nnModules import BaseTask


class SimpleClassifier(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SimpleClassifier, self).__init__()
        self.dim = dim
        if backbone:
            self.backbone = backbone
        if neck:
            self.neck = neck
        if head:
            self.head = head

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def forward_train(self, data_batch: dict, *args, **kwargs):
        img = data_batch['img']
        label = data_batch['gt_cls'][..., 2 * self.dim]
        x = self.extract_feat(img)
        outs = self.head(x)
        
        rescale = 2 * (1 - 0.5 ** len(outs))
        # print(len(outs))
        losses = OrderedDict()
        for i in range(len(outs)):
            prediction = outs[i]
            gt = label.squeeze()
            self.try_to_info("prediction", prediction.shape, gt.shape)
            
            loss_dict = self.head.loss(prediction, gt.long())
            reference = self.head.loss_reference(prediction, gt.long())
            
            assert isinstance(loss_dict, dict), f"head.loss must return a dict of losses but got {type(loss_dict)}"
            assert isinstance(reference, torch.Tensor), "head.base_loss must return a tensor"
            
            losses['reference'] = losses.get('reference', 0) + reference / (2 ** i) / rescale
            for loss_name, loss_value in loss_dict.items():
                losses[loss_name] = losses.get(loss_name, 0) + loss_value / (2 ** i) / rescale
            
            # print(reference, loss_dict, losses)
        return losses
    
    def forward_infer(self, data_batch: dict, *args, **kwargs):
        img = data_batch['img']
        x = self.extract_feat(img)
        outs = self.head(x)
        return outs[0]
