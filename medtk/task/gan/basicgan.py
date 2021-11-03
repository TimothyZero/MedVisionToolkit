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


class SimpleGAN(BaseTask):
    def __init__(self,
                 dim,
                 generator=None,
                 discriminator=None):
        super(SimpleGAN, self).__init__()
        self.dim = dim
        if generator:
            self.generator = generator
        if discriminator:
            self.discriminator = discriminator

    # def extract_feat(self, img):
    #     """Directly extract features from the backbone+neck
    #     """
    #     x = self.backbone(img)
    #     return x
    #
    # def forward_train(self, img_meta, img):
    #     # Sample noise as generator input
    #     z = Variable(torch.from_numpy(np.random.normal(0, 1, (img.shape[0], self.generator.latent_dim))))
    #     gen_imgs = self.extract_feat(z)
    #     outs = self.head(x)
    #     # print(len(outs))
    #
    #     losses = OrderedDict()
    #     prediction = outs[0]
    #
    #     loss_dict = self.head.loss(prediction, gt.long())
    #     reference = self.head.loss_reference(prediction, gt.long())
    #
    #     assert isinstance(loss_dict, dict), "head.loss must return a dict of losses"
    #     assert isinstance(reference, torch.Tensor), "head.base_loss must return a tensor"
    #
    #     losses['reference'] = reference
    #     losses.update(loss_dict)
    #     return losses, outs[0]
    #
    # def forward_infer(self, img_meta, img):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     return outs[0]
    #
    # def forward(self, batch, return_loss=True):
    #     # print(self.device)
    #     assert all([i in batch.keys() for i in ['img_meta', 'img']]), print(batch.keys())
    #     # print(batch['img_meta']['img_dim'])
    #
    #     img = batch['img']
    #     # print(img.shape)
    #     assert torch.max(img) <= 1.0 and torch.min(img) >= -1.0, \
    #         'max={} min={}'.format(torch.max(img).item(), torch.min(img).item())
    #
    #     # print(torch.min(img), torch.max(img), img.shape)
    #     if not return_loss:
    #         return self.forward_infer(batch['img_meta'], img.to(self.device))
    #     else:
    #         return self.forward_train(batch['img_meta'], img.to(self.device))
