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


def nmsNd_numpy(dets: np.ndarray, threshold: float):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :return: the rest ids of dets
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3), dets.shape

    scores = dets[:, -1].astype(np.float32).copy()
    bboxes = dets[:, :-1].astype(np.float32).copy()
    assert bboxes.shape[-1] == 2 * dim

    area = np.prod(bboxes[:, dim:] - bboxes[:, :dim] + 1, axis=-1)
    # print(area)

    order = scores.argsort()[::-1]

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)

        overlap = np.minimum(bboxes[i, dim:], bboxes[order[1:]][:, dim:])
        overlap = overlap - np.maximum(bboxes[i, :dim], bboxes[order[1:]][:, :dim]) + 1
        overlap = np.maximum(overlap, 0)
        inter = np.prod(overlap, axis=-1)
        # print(inter)

        union = area[i] + area[order[1:]] - inter
        iou = inter / union
        # print(iou)

        index = np.where(iou <= threshold)[0]
        # print(index)

        # similar to soft nmsNd_cuda
        # weight = np.exp(-(iou * iou) / 0.5)
        # scores[order[1:]] = weight * scores[order[1:]]

        order = order[index + 1]

    dets = np.concatenate((bboxes, scores[:, None]), axis=1)
    keep = np.array(keep)
    return keep, dets


if __name__ == "__main__":
    import numpy as np
    import time

    dets_2d = np.array([[49.1, 32.4, 51.0, 35.9, 0.91],
                        [49.3, 32.9, 51.0, 35.3, 0.9],
                        [49.2, 31.8, 51.0, 35.4, 0.52],
                        [35.1, 11.5, 39.1, 15.7, 0.51],
                        [35.6, 11.8, 39.3, 14.2, 0.5],
                        [35.3, 11.5, 39.9, 14.5, 0.4],
                        [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
    dets_3d = np.array([[49.1, 32.4, 23.0, 51.0, 35.9, 25.0, 0.91],
                        [49.3, 32.9, 33.0, 51.0, 35.3, 35.0, 0.9],
                        [49.2, 31.8, 23.0, 51.0, 35.4, 25.0, 0.52],
                        [35.1, 11.5, 23.0, 39.1, 15.7, 25.0, 0.51],
                        [35.6, 11.8, 23.0, 39.3, 14.2, 25.0, 0.5],
                        [35.3, 11.5, 23.0, 39.9, 14.5, 25.0, 0.4],
                        [35.2, 11.7, 23.0, 39.7, 15.7, 25.0, 0.3]], dtype=np.float32)

    iou_thr = 0.7

    start = time.time()
    index, suppressed = nmsNd_numpy(dets_2d, iou_thr)
    print(time.time() - start)
    print(index)
    print(suppressed[index])

