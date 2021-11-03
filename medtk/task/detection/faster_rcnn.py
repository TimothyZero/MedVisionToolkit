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

from medtk.model.nnModules import BaseTask


class FasterRCNN(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None):
        super(FasterRCNN, self).__init__()
        self.dim = dim
        if backbone:
            self.backbone = backbone
        if neck:
            self.neck = neck
        if rpn_head:
            self.rpn_head = rpn_head
        if roi_head:
            self.roi_head = roi_head

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)

        feats = self.extract_feat(img)

        rpn_loss_dict, batch_proposals, net_output = self.rpn_head.forward_train(feats, gt_labels, gt_bboxes)

        roi_loss_dict = self.roi_head.forward_train(feats, batch_proposals, gt_labels, gt_bboxes)

        losses = {}
        losses.update(rpn_loss_dict)
        losses.update(roi_loss_dict)

        return losses, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        feats = self.extract_feat(img)

        rpn_loss_dict, batch_proposals, net_output = self.rpn_head.forward_valid(feats, gt_labels, gt_bboxes)

        roi_loss_dict, batch_bboxes = self.roi_head.forward_valid(feats, batch_proposals, gt_labels, gt_bboxes)

        metrics = self.metric(data_batch, batch_bboxes)

        metrics_losses = {}
        metrics_losses.update(metrics)
        metrics_losses.update(rpn_loss_dict)
        metrics_losses.update(roi_loss_dict)

        return metrics_losses, batch_bboxes, net_output

    def forward_infer(self, data_batch, rpn=False, *args, **kwargs):
        img = data_batch['img']

        feats = self.extract_feat(img)

        if not rpn:
            batch_proposals, batch_anchor_id, net_output = self.rpn_head.forward_infer(feats)
            batch_bboxes, batch_anchor_id = self.roi_head.forward_infer(feats, batch_proposals, batch_anchor_id)
        else:
            print('rpn')
            self.rpn_head.nms = dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.15,
                nms_fun=dict(type='nms', iou_threshold=0.1),
                max_per_img=100)
            self.rpn_head.proposal_cfg = None
            batch_bboxes, batch_anchor_id, net_output = self.rpn_head.forward_infer(feats)

        return batch_bboxes, net_output

    def metric(self, data_batch: dict, net_output):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']

        self.try_to_info("roi_results", net_output)
        one_metric = self.roi_head.metric(net_output, label)
        metrics = {}
        for k, v in one_metric.items():
            metrics['roi_' + k] = v
        return metrics


# if __name__ == "__main__":
#     import numpy as np
#
#     backbone = dict(
#         type="UNetEncoder",
#         in_channels=1,
#         base_width=16,
#         out_indices=[0, 1, 2, 3, 4]
#     )
#     neck = dict(
#         type="FPN",
#         in_channels=[16, 32, 64, 128, 256],
#         out_channels=16,
#         start_level=0,
#         add_extra_convs=True,
#         num_outs=5,
#         out_indices=(0, 1),
#     )
#     rpn_head = dict(
#         type="RPNHead",
#         in_channels=16,
#         base_scales=4,
#         scales=[4, 6, 7],
#         ratios=[1]
#     )
#     bbox_head = dict(
#         type="RcnnHead",
#         feat_channels=16,
#         num_classes=2
#     )
#
#     model = FasterRCNN(2, backbone, neck, rpn_head, bbox_head)
#     model.setLog()
#
#     batch = 2
#     image = torch.rand((batch, 1, 32, 64))
#     # gra = torch.rand((2, 5, 64, 64))
#     # label = torch.ones((batch, 32, 32))
#
#     gt_bboxes = np.array([[0, 0, 4.2, 4.2],
#                           [0, 1.1, 5, 5.2],
#                           [3, 3, 8, 8]])
#
#     gt_labels = np.array([1, 1, 1])
#     label = (torch.stack([torch.tensor(gt_bboxes).float()] * batch, dim=0),
#              torch.stack([torch.tensor(gt_labels).float()] * batch, dim=0))
#     # print(label[0].shape, label[1].shape)
#
#     losses, net_output = model.forward_train(None, image, label)
#     # self.try_to_log(losses)
#
#     loss_total = 0
#     for name, loss in losses.items():
#         print(name, loss)
#         loss_total = loss_total + loss
#
#     print("backward...")
#     loss_total.backward()
#
#     """ view roi extractor"""
#     # self.try_to_log("\n\n\n\nrois", rois[-1])
#     # plt.imshow(feats[0][1].mean(dim=0).detach().cpu().numpy())
#     # plt.colorbar()
#     # plt.show()
#     #
#     # plt.imshow(roi_features[-1].mean(dim=0).detach().cpu().numpy())
#     # plt.colorbar()
#     # plt.show()
#     #
#     # _, x1, y1, x2, y2 = rois[-1]
#     # print(int(y1//2), int(y2//2), int(x1//2), int(x2//2))
#     # img = feats[0][1].mean(dim=0).detach().cpu().numpy()
#     # print(img.shape)
#     # plt.imshow(img[int(y1//2):int(y2//2), int(x1//2):int(x2//2)])
#     # plt.colorbar()
#     # plt.show()
