import torch.nn as nn
import torch

from ..nnModules import ComponentModule
from ..nd import ConvNd, BatchNormNd


class Generator(ComponentModule):
    def __init__(self, dim, latent_dim, img_size, channels=1):
        super(Generator, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = channels

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 3))

        self.conv_blocks = nn.Sequential(
            BatchNormNd(self.dim)(128),
            nn.Upsample(scale_factor=2),
            ConvNd(self.dim)(128, 128, 3, stride=1, padding=1),
            BatchNormNd(self.dim)(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            ConvNd(self.dim)(128, 64, 3, stride=1, padding=1),
            BatchNormNd(self.dim)(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvNd(self.dim)(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ConvNd(self.dim)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, BatchNormNd(self.dim)):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        # base = getSphereStructure3D(self.img_size, 2 * (self.img_size // 2 - 3) + 1)
        # img = img + torch.from_numpy(base).float().cuda()
        return img
