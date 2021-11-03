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


def softnmsNd_numpy(dets: np.ndarray, threshold: float, method=1, sigma=0.5):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :param method
    :param sigma: gaussian filter sigma
    :return: the index of the selected boxes
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3)

    scores = dets[:, -1].copy()
    bboxes = dets[:, :-1].copy()
    assert bboxes.shape[-1] == 2 * dim

    N = bboxes.shape[0]
    index = np.array([np.arange(N)])
    bboxes = np.concatenate((bboxes, index.T), axis=1)

    area = np.prod(bboxes[:, dim:-1] - bboxes[:, :dim] + 1, axis=-1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tmp_score = scores[i].copy()
        pos = i + 1

        if i != N - 1:
            max_score = np.max(scores[pos:], axis=0)
            max_pos = np.argmax(scores[pos:], axis=0)
            if tmp_score < max_score:
                bboxes[i], bboxes[max_pos.item() + i + 1] = bboxes[max_pos.item() + i + 1].copy(), bboxes[i].copy()
                scores[i], scores[max_pos.item() + i + 1] = scores[max_pos.item() + i + 1].copy(), scores[i].copy()
                area[i], area[max_pos + i + 1] = area[max_pos + i + 1].copy(), area[i].copy()

        overlap = np.minimum(bboxes[i, dim:-1], bboxes[pos:, dim:-1])
        overlap = overlap - np.maximum(bboxes[i, :dim], bboxes[pos:, :dim]) + 1
        overlap = np.maximum(overlap, 0)
        inter = np.prod(overlap, axis=-1)

        union = area[i] + area[pos:] - inter
        iou = inter / union

        if method == 1:
            # linear
            weight = np.where(iou > threshold, 1 - iou, np.ones_like(iou))
        else:
            # Gaussian decay
            weight = np.exp(-(iou * iou) / sigma)
        scores[pos:] = weight * scores[pos:]

    keep = bboxes[:, -1].copy().astype(np.int)
    bboxes[:, -1] = scores
    return keep, bboxes
