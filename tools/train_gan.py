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
import torch
from torch.autograd import Variable

from medtk.task import SimpleGAN
from medtk.data import BasicPairDataset, ImageIO, vtkScreenshot
from medtk.runner.checkpoint import save_checkpoint

latent_dim = 100

s = SimpleGAN(
    dim=3,
    generator=dict(
        type="Generator",
        latent_dim=latent_dim,
        img_size=32
    ),
    discriminator=dict(
        type="Discriminator",
        img_size=32,
    ),
    train_cfg=None,
    test_cfg=None
).cuda()
# print(s)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = s.generator
discriminator = s.discriminator

dataset = BasicPairDataset("/media/timothy/Datasets/Classification/Nodule/train_dataset.json",
                           "/media/timothy/Datasets/Classification/Nodule/",
                           pipeline=[
                               dict(type='LoadImageFromFile'),
                               dict(type='LoadAnnotations', with_cls=True),
                               # dict(type='RandomFlipND', p=0.5),
                               dict(type='RandomGamma', p=0.5, gamma=0.2),
                               # dict(type='RandomRotateND', p=0.5, angle=30),
                               dict(type='NormalizeND', mean=(-400,), std=(750,), clip=True),
                               dict(type='Collect', keys=['img', 'gt_cls']),
                           ])
# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ----------
#  Training
# ----------
n_epochs = 500

Tensor = torch.cuda.FloatTensor
for epoch in range(n_epochs):
    for i, data in enumerate(dataloader):

        # Adversarial ground truths
        imgs = data['img']
        valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0).to(s.device), requires_grad=False)
        fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0).to(s.device), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % 400 == 0:
            vtkScreenshot(f"/media/timothy/Datasets/Classification/Nodule/valid_results/{batches_done}.png", gen_imgs[0, 0].cpu().detach().numpy())
            ImageIO.saveArray(f"/media/timothy/Datasets/Classification/Nodule/valid_results/{batches_done}.nii.gz",
                              gen_imgs[0, 0].cpu().detach().numpy())

            save_checkpoint(s, "/media/timothy/Datasets/Classification/Nodule/gan.pth")