import torch
from torch import nn

from medtk.model.nnModules import BlockModule, ComponentModule


class PostRes(BlockModule):
    def __init__(self, n_in, n_out, stride=1, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(PostRes, self).__init__(None, norm_cfg, act_cfg)
        self.conv1 = self.build_conv(3, n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = self.build_norm(3, n_out)
        self.relu = self.build_act(inplace=True)
        self.conv2 = self.build_conv(3, n_out, n_out, kernel_size=3, padding=1, conv_cfg=conv_cfg)
        self.bn2 = self.build_norm(3, n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                self.build_conv(3, n_in, n_out, kernel_size=1, stride=stride),
                self.build_norm(3, n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class DeepLungBK(ComponentModule):
    def __init__(self,
                 dim,
                 with_coord=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 stages_with_dcn=(False, False, False, False),
                 dcn_cfg=None):
        super(DeepLungBK, self).__init__(conv_cfg, norm_cfg, act_cfg)
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.dim = dim
        assert dim == 3
        self.with_coord = with_coord
        self.preBlock = nn.Sequential(
            self.build_conv(self.dim, 1, 24, kernel_size=3, padding=1),
            self.build_norm(self.dim, 24),
            self.build_act(inplace=True),
            self.build_conv(self.dim, 24, 24, kernel_size=3, padding=1),
            self.build_norm(self.dim, 24),
            self.build_act(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            dcn = dcn_cfg if stages_with_dcn[i] else None
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(
                        self.featureNum_forw[i],
                        self.featureNum_forw[i + 1],
                        1,
                        dcn,
                        norm_cfg,
                        act_cfg))
                else:
                    blocks.append(PostRes(
                        self.featureNum_forw[i + 1],
                        self.featureNum_forw[i + 1],
                        1,
                        conv_cfg,
                        norm_cfg,
                        act_cfg))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3 if self.with_coord else 0
                    else:
                        addition = 0
                    blocks.append(PostRes(
                        self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                        self.featureNum_back[i],
                        1,
                        conv_cfg,
                        norm_cfg,
                        act_cfg))
                else:
                    blocks.append(PostRes(
                        self.featureNum_back[i],
                        self.featureNum_back[i],
                        1,
                        conv_cfg,
                        norm_cfg,
                        act_cfg))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            self.build_norm(3, 64),
            self.build_act(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            self.build_norm(3, 64),
            self.build_act(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        features = self._enc_(x)
        return self._dec_(*features)

    def _enc_(self, x):
        out = self.preBlock(x)
        out1 = self.forw1(self.maxpool(out))
        out2 = self.forw2(self.maxpool(out1))
        out3 = self.forw3(self.maxpool(out2))
        out4 = self.forw4(self.maxpool(out3))
        return [out2, out3, out4]

    def _dec_(self, out2, out3, out4):
        rev3 = self.path1(out4)
        rev3 = self.back3(torch.cat((rev3, out3), 1))
        # rev3 = self.drop(rev3)

        rev2 = self.path2(rev3)
        if self.with_coord:
            coord = torch.stack(torch.meshgrid(*map(torch.arange, out2.shape[2:])))
            coord = coord.repeat(out2.shape[0], 1, 1, 1, 1).to(out2.device)
            rev2 = self.back2(torch.cat((rev2, out2, coord), 1))
        else:
            rev2 = self.back2(torch.cat((rev2, out2), 1))
        rev2 = self.drop(rev2)

        return [rev2]


if __name__ == "__main__":
    net = DeepLungBK(3,
                     # with_coord=True,
                     # stages_with_dcn=(True, True, False, False),
                     # dcn_cfg=dict(type='DCNv1'),
                     # norm_cfg=dict(type='InstanceNorm')
                     ).cuda()
    print(net)
    net.print_model_params()
    x = torch.rand((1, 1, 96, 96, 96)).cuda()
    data = {
        'img':       x,
        'img_meta':  None,
        'gt_labels': 2 * torch.rand((1, 24, 24, 24, 3, 5)).cuda() - 1,
        'gt_bboxes': torch.ones((1, 3, 24, 24, 24)).cuda(),
    }
    features = net(data['img'])
    [print(i.shape) for i in features]
