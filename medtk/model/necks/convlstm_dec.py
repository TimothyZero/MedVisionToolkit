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

import torch.nn as nn
import torch

from medtk.model.blocks import ConvNormAct
from medtk.model.nnModules import ComponentModule, BlockModule


class ConvLSTMCell(BlockModule):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                            out_channels=4 * self.hidden_dim,  # important, will spilt into 4
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias),
                                  nn.BatchNorm2d(4 * self.hidden_dim),
                                  nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                           out_channels=self.hidden_dim,  # important, will spilt into 4
                                           kernel_size=self.kernel_size,
                                           padding=self.padding,
                                           bias=self.bias),
                                 nn.BatchNorm2d(self.hidden_dim),
                                 nn.ReLU(inplace=True))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        y_next = torch.sigmoid(self.out(h_next))

        return y_next, (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device))


class UpConvLSTMCell(BlockModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(UpConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                            out_channels=4 * self.hidden_dim,  # important, will spilt into 4
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            bias=self.bias),
                                  nn.BatchNorm2d(4 * self.hidden_dim),
                                  nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim,
                                           out_channels=self.hidden_dim,  # important, will spilt into 4
                                           kernel_size=self.kernel_size,
                                           padding=self.padding,
                                           bias=self.bias),
                                 nn.BatchNorm2d(self.hidden_dim),
                                 nn.ReLU(inplace=True))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        input_tensor = torch.nn.functional.interpolate(input_tensor, scale_factor=2)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        y_next = torch.sigmoid(self.out(h_next))

        return y_next, (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height * 2, width * 2, device=self.conv[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height * 2, width * 2, device=self.conv[0].weight.device))


class ConvLSTMDecoder(ComponentModule):
    """

    Parameters:
        in_channels: Number of channels in input

    Input:
        A tensor of size B, C, D, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:

    """

    def __init__(self,
                 dim,
                 in_channels,
                 out_indices,
                 bidirectional=False):
        super(ConvLSTMDecoder, self).__init__()
        assert isinstance(in_channels, (tuple, list)), 'in_channels must be a list or tuple'
        self.dim = dim
        assert dim == 3, 'only support 3d image'
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.stages = len(in_channels) - 1
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        assert max(self.out_indices) + 1 <= len(in_channels)

        up_cells = []
        for i, planes in enumerate(self.in_channels[-2::-1]):
            # print("enc cell", i, "==>", planes)
            up_cells.extend([UpConvLSTMCell(input_dim=planes,
                                            hidden_dim=planes,
                                            kernel_size=3,
                                            bias=True)] * self.num_directions)

        self.convLSTMCells = nn.ModuleList(up_cells)
        # print(self.convLSTMCells)

    def _init_hidden(self, batch_size, h, w):
        init_states = []
        # out = self.convLSTMCells[0].init_hidden(batch_size, (h, w))
        # init_states.append(out)
        for i in range(self.stages):
            for num in range(self.num_directions):
                c = self.convLSTMCells[i * self.num_directions + num]
                out = c.init_hidden(batch_size, (h // 2 ** (self.stages - i), w // 2 ** (self.stages - i)))
                init_states.append(out)
        return init_states

    def forward(self, x, hidden_state=None):
        """

        Parameters
        ----------
        x: todo
            5-D Tensor (b, c, d, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # input is # b, c, d, h, w ==>  # b, d, c, h, w
        x = [i.permute(0, 2, 1, 3, 4) for i in x]
        b, _, _, h, w = x[0].size()
        # [print(i.shape) for i in x]

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(b, h, w)

        input_layers = [x[-1]]
        # last_state_list = []

        for layer_idx in range(self.stages):
            cur_output_layer = []
            cur_input_layer = input_layers[-1].clone()

            b, d, c, h, w = cur_input_layer.shape
            self.try_to_info("before reshape", cur_input_layer.shape)
            cur_input_layer = cur_input_layer.reshape((b, d * 2, -1, h, w))
            self.try_to_info("after  reshape", cur_input_layer.shape)

            self.try_to_info("layer_idx", layer_idx, cur_input_layer.shape)

            num_slices = cur_input_layer.shape[1]
            for direction in range(self.num_directions):
                tmp = cur_input_layer.clone()
                cell_idx = layer_idx * self.num_directions + direction
                hidden, cell = hidden_state[cell_idx]
                convLSTMCell = self.convLSTMCells[cell_idx]
                op = -1 if direction == 1 else 1
                for t in range(num_slices)[::op]:
                    out, (hidden, cell) = convLSTMCell(input_tensor=tmp[:, t, :, :, :],
                                                       cur_state=[hidden, cell])
                    cur_output_layer.append(out)
                    self.try_to_info(f"layer_idx {layer_idx} - cell_idx {cell_idx}", f'slice ={t}', 'y, h, c',
                                     out.shape, hidden.shape, cell.shape)

            if self.bidirectional:
                cur_output_layer_f = torch.stack(cur_output_layer[:num_slices], dim=1)
                cur_output_layer_b = torch.stack(cur_output_layer[-1:-num_slices - 1:-1], dim=1)
                self.try_to_info(f"layer_idx {layer_idx + 1} cur_output_layer forward and backward",
                                 cur_output_layer_f.shape, cur_output_layer_b.shape)
                cur_output_layer = cur_output_layer_f + cur_output_layer_b
            else:
                cur_output_layer = torch.stack(cur_output_layer, dim=1)

            self.try_to_info(f"layer_idx {layer_idx + 1} cur_output_layer", cur_output_layer.shape)
            input_layers.append(cur_output_layer + x[-layer_idx - 2])
            # last_state_list.append([hidden, cell])

        for i in range(len(input_layers)):
            # b, d, c, h, w => b, c, d, h, w
            input_layers[i] = input_layers[i].permute(0, 2, 1, 3, 4)

        input_layers.pop(0)
        input_layers.reverse()
        # return input_layers
        return [input_layers[i] for i in self.out_indices]  # , last_state_list


if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss()
    x = [torch.rand((2, 3, 32, 64, 64)),
         torch.rand((2, 6, 16, 32, 32)),
         torch.rand((2, 12, 8, 16, 16)),
         torch.rand((2, 24, 4, 8, 8)),
         torch.rand((2, 48, 2, 4, 4))
         ]
    convlstm = ConvLSTMDecoder(dim=3, in_channels=[3, 6, 12, 24, 48], out_indices=(0,), bidirectional=False)
    # print(convlstm)
    # print(convlstm.modules)
    convlstm.set_log()

    layer_output_list = convlstm(x)
    print("")
    for i in range(len(layer_output_list)):
        print(layer_output_list[i].shape)
