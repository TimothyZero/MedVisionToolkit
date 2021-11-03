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
#

from medtk.model.heads_det.roi_heads.base_roi_head import BaseRoIHead


class ROIHead(BaseRoIHead):
    def __init__(self,
                 dim=None,
                 bbox_roi_extractor=None,
                 bbox_head=None):
        super(ROIHead, self).__init__()

        self.bbox_roi_extractor = bbox_roi_extractor

        self.bbox_head = bbox_head
        self.bbox_head.set_extractor(self.bbox_roi_extractor)

    def forward_train(self,
                      multi_level_features,
                      batch_proposals,
                      batch_gt_labels,
                      batch_gt_bboxes):
        return self.bbox_head.forward_train(multi_level_features,
                                            batch_proposals,
                                            batch_gt_labels,
                                            batch_gt_bboxes)

    def forward_valid(self,
                      multi_level_features,
                      batch_proposals,
                      batch_gt_labels,
                      batch_gt_bboxes):
        return self.bbox_head.forward_valid(multi_level_features,
                                            batch_proposals,
                                            batch_gt_labels,
                                            batch_gt_bboxes)

    def forward_infer(self,
                      multi_level_features,
                      batch_proposals):
        return self.bbox_head.forward_infer(multi_level_features,
                                            batch_proposals)

    def metric(self, results, ground_truth):
        return self.bbox_head.metric(results, ground_truth)
