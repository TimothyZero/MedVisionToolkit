import torch
from torch import nn

from medtk.model.nnModules import BlockModule, ComponentModule


class PostRes(BlockModule):
    def __init__(self, n_in, n_out, stride=1, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(PostRes, self).__init__(None, norm_cfg, act_cfg)
        self.conv1 = self.build_conv(3, n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = self.build_norm(3, n_out)
        self.relu = self.build_act(self.act_cfg)
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


class DeepLungEnc(ComponentModule):
    def __init__(self,
                 dim,
                 in_channels: int,
                 features=(32, 64, 64, 64),
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 num_blocks=(2, 2, 3, 3),
                 out_indices=(0, 1, 2, 3),
                 first_conv=(24, 3, 1),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 stages_with_dcn=(False, False, False, False),
                 dcn_cfg=None):
        super(DeepLungEnc, self).__init__(conv_cfg, norm_cfg, act_cfg)
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.dim = dim
        assert dim == 3
        self.stages = len(features)
        self.out_indices = out_indices
        first_features, first_kernel, first_stride = first_conv
        self.preBlock = nn.Sequential(
            self.build_conv(self.dim, in_channels, first_features,
                            kernel_size=first_kernel, stride=first_stride, padding=first_kernel//2),
            self.build_norm(self.dim, first_features),
            self.build_act(self.act_cfg),
            self.build_conv(self.dim, first_features, first_features, kernel_size=3, padding=1),
            self.build_norm(self.dim, first_features),
            self.build_act(self.act_cfg))

        self.featureNum_forw = [first_features] + list(features)
        for i in range(len(num_blocks)):
            dcn = dcn_cfg if stages_with_dcn[i] else None
            blocks = []
            for j in range(num_blocks[i]):
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

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.preBlock(x)

        outs = []
        for i in range(self.stages):
            layer_name = 'forw{}'.format(i + 1)
            layer = getattr(self, layer_name)
            x = layer(self.maxpool(x))
            if i in self.out_indices:
                outs.append(x)
        return outs


if __name__ == "__main__":
    net = DeepLungEnc(dim=3,
                      in_channels=1,
                      out_indices=(0, 1, 2, 3)
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
        'img_meta':  None
    }
    features = net(data['img'])
    [print(i.shape) for i in features]
