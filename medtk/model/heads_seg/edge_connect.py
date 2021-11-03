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

# -*- coding:utf-8 -*-
import os
import torch.optim as optim

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _ntuple
from typing import Union, Tuple

from torch.nn.modules.padding import ReflectionPad2d as OriReflectionPad2d


def prepare_onnx_paddings(dim, pad):
    assert isinstance(dim, int)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    assert len(pad) <= dim * 2
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    assert len(paddings) == dim * 2
    return paddings


def flip(x: Variable, dim: int) -> Variable:
    """Flip torch Variable along given dimension axis."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1),
                ('cpu', 'cuda')[x.is_cuda])().long(), :]
    x = x.view(xsize)
    return x


class ReflectionPad2d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""

    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad2d, self).__init__()
        self.padding = _ntuple(4)(padding)

    def forward(self, input: Variable) -> Variable:
        return ReflectionPadNd.apply(input, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + str(self.padding) + ')'


class ReflectionPad3d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""

    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input: Variable) -> Variable:
        return ReflectionPadNd.apply(input, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + str(self.padding) + ')'


class ReflectionPadNd(Function):
    """Padding for same convolutional layer."""

    @staticmethod
    def symbolic(g, input: Variable, padding: Union[int, Tuple[int]]):
        paddings = prepare_onnx_paddings(len(input.type().sizes()), padding)
        return g.op("Pad", input, pads_i=paddings, mode_s="reflect")

    @staticmethod
    def forward(ctx: Function, input: Variable, pad: Tuple[int]) -> Variable:
        ctx.pad = pad
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b)
                             for a, b in zip(pad[:-1:2], pad[1::2])]
                            [::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i])
                         for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # Create output tensor by concatenating with reflected chunks.
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).zero_()
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                chunk1 = flip(c_input.narrow(i, 1, pad[0]), i)
                c_input = torch.cat((chunk1, c_input), i)
            if p[1] > 0:
                chunk2 = flip(c_input.narrow(i, c_input.shape[i] - p[1] - 1, p[1]), i)
                c_input = torch.cat((c_input, chunk2), i)
        output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Variable) -> Variable:
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x, ) for x in ctx.input_size]

        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0],
                                             cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0,
                                             cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm3d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            ReflectionPad3d(3),
            spectral_norm(nn.Conv3d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm3d(64, track_running_stats=False),
            nn.ReLU(True),

            ReflectionPad3d(3),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            ReflectionPad3d(dilation),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
            nn.ReLU(True),

            ReflectionPad3d(1),
            spectral_norm(nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm3d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        # plt.imshow(images[0, 0].cpu().numpy())
        # plt.show()
        # plt.imshow(edges_masked[0, 0].cpu().numpy())
        # plt.show()
        # plt.imshow(images_masked[0, 0].cpu().numpy())
        # plt.show()
        # raise
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward(retain_graph=True)
        if gen_loss is not None:
            gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()


DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'MASK': 3,                      # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'EDGE': 1,                      # 1: canny, 2: external
    'NMS': 1,                       # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'LR': 0.0001,                   # learning rate
    'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 8,                # input batch size for training
    'INPUT_SIZE': 256,              # input image size for training 0 for original size
    'SIGMA': 2,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model

    'EDGE_THRESHOLD': 0.5,          # edge detection threshold
    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 1,         # style loss weight
    'CONTENT_LOSS_WEIGHT': 1,       # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# adversarial loss weight

    'GAN_LOSS': 'nsgan',            # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,             # fake images pool size

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}


class Config(dict):
    def __init__(self):
        self._dict = DEFAULT_CONFIG
        self._dict['PATH'] = "/home/timothy/PycharmProjects/MedToolkit/work_dirs"

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._dict)
        print('')
        print('---------------------------------')
        print('')


a = EdgeModel(Config())
print(a)

images, edges, masks = torch.zeros((1, 1, 32, 32, 32)), torch.zeros((1, 1, 32, 32, 32)), torch.zeros((1, 1, 32, 32, 32))
outputs, gen_loss, dis_loss, logs = a.process(images, edges, masks)
print(outputs.shape)
print(gen_loss, dis_loss)

gen_loss.backward()
dis_loss.backward()