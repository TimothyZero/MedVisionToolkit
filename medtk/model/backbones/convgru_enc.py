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

from medtk.model.nnModules import BlockModule, ComponentModule


class ConvGRUCell(BlockModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvGRU cell.

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

        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.reset_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                  out_channels=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  bias=self.bias),
                                        nn.BatchNorm2d(self.hidden_dim),
                                        nn.ReLU(inplace=True))
        self.update_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                   out_channels=self.hidden_dim,
                                                   kernel_size=self.kernel_size,
                                                   padding=self.padding,
                                                   bias=self.bias),
                                         nn.BatchNorm2d(self.hidden_dim),
                                         nn.ReLU(inplace=True))
        self.out_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                out_channels=self.hidden_dim,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                bias=self.bias),
                                      nn.BatchNorm2d(self.hidden_dim),
                                      nn.ReLU(inplace=True))

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state

        stacked_inputs = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        update = torch.sigmoid(self.update_gate(stacked_inputs))

        h_p = torch.tanh(self.out_gate(torch.cat([input_tensor, reset * h_cur], dim=1)))
        h_next = (1 - update) * h_cur + update * h_p

        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.out_gate[0].weight.device)


class DownConvGRUCell(BlockModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvGRU cell.

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

        super(DownConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.reset_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                  out_channels=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding,
                                                  bias=self.bias),
                                        nn.BatchNorm2d(self.hidden_dim),
                                        nn.ReLU(inplace=True))
        self.update_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                   out_channels=self.hidden_dim,
                                                   kernel_size=self.kernel_size,
                                                   padding=self.padding,
                                                   bias=self.bias),
                                         nn.BatchNorm2d(self.hidden_dim),
                                         nn.ReLU(inplace=True))
        self.out_gate = nn.Sequential(nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                                out_channels=self.hidden_dim,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                bias=self.bias),
                                      nn.BatchNorm2d(self.hidden_dim),
                                      nn.ReLU(inplace=True))

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state

        input_tensor = torch.nn.functional.avg_pool2d(input_tensor, 2)
        stacked_inputs = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        update = torch.sigmoid(self.update_gate(stacked_inputs))

        h_p = torch.tanh(self.out_gate(torch.cat([input_tensor, reset * h_cur], dim=1)))
        h_next = (1 - update) * h_cur + update * h_p

        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height // 2, width // 2, device=self.out_gate[0].weight.device)


class ConvGRUEncoder(ComponentModule):
    """

    Parameters:
        dim: dimension of image, 2 or 3
        in_channels: Number of channels in input
        base_width: base num of feature maps
        out_indices: out indices of outs
        stages: Number of GRU layers stacked on each other
        bidirectional: if bidirectional, true or false

    Input:
        A tensor of size B, C, D, H, W
    Output:

    Example:

    """

    def __init__(self, dim, in_channels, base_width=16, stages=4, out_indices=(0, 1, 2, 3, 4), bidirectional=False):
        super(ConvGRUEncoder, self).__init__()

        self.dim = dim
        assert dim == 3, 'only support 3d image'
        self.in_channels = in_channels
        self.stages = stages
        self.out_indices = out_indices
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        assert max(self.out_indices) <= self.stages

        in_cell = ConvGRUCell(input_dim=in_channels,
                              hidden_dim=base_width,
                              kernel_size=3,
                              bias=True)

        cells = []
        cells.extend([in_cell] * self.num_directions)
        for i in range(self.stages):
            planes = base_width * pow(2, i + 1)
            cells.extend([DownConvGRUCell(input_dim=planes,
                                          hidden_dim=planes,
                                          kernel_size=3,
                                          bias=True)] * self.num_directions)
        self.convGRUCells = nn.ModuleList(cells)
        self.try_to_info(len(self.convGRUCells))

    def _init_hidden(self, batch_size, h, w):
        init_states = []
        for num in range(self.num_directions):
            out = self.convGRUCells[num].init_hidden(batch_size, (h, w))
            init_states.append(out)

        for i in range(self.stages):
            for num in range(self.num_directions):
                c = self.convGRUCells[i * self.num_directions + num + self.num_directions]
                out = c.init_hidden(batch_size, (h // 2 ** i, w // 2 ** i))
                init_states.append(out)

        return init_states

    def forward(self, input_tensor, hidden_state=None):
        # input is # b, c, d, h, w ==>  # b, d, c, h, w
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # Since the init is done in forward. Can send image size here
        hidden_state = self._init_hidden(b, h, w)

        input_layers = [input_tensor]
        # last_state_list = []

        # layer 0
        cur_output_layer = []
        cur_input_layer = input_layers[-1].clone()

        self.try_to_info("layer_idx", 0, cur_input_layer.shape)

        num_slices = cur_input_layer.shape[1]
        for direction in range(self.num_directions):
            tmp = cur_input_layer.clone()
            cell_idx = direction
            hidden = hidden_state[cell_idx]
            convGRUCell = self.convGRUCells[cell_idx]
            op = -1 if direction == 1 else 1
            for t in range(num_slices)[::op]:
                hidden = convGRUCell(input_tensor=tmp[:, t, :, :, :],
                                     cur_state=hidden)
                cur_output_layer.append(hidden)
                self.try_to_info(f"layer_idx {0} - cell_idx {cell_idx}", f'slice ={t}', 'y, h, c',
                                 hidden.shape)
        if self.bidirectional:
            cur_output_layer_f = torch.stack(cur_output_layer[:num_slices], dim=1)
            cur_output_layer_b = torch.stack(cur_output_layer[-1:-num_slices - 1:-1], dim=1)
            self.try_to_info(f"layer_idx {0} cur_output_layer forward and backward",
                             cur_output_layer_f.shape, cur_output_layer_b.shape)
            cur_output_layer = cur_output_layer_f + cur_output_layer_b
        else:
            cur_output_layer = torch.stack(cur_output_layer, dim=1)

        self.try_to_info(f"layer_idx {0} cur_output_layer", cur_output_layer.shape)
        input_layers.append(cur_output_layer)

        # last_state_list.append([hidden, cell])

        # layer 2...
        for layer_idx in range(self.stages):
            cur_output_layer = []
            cur_input_layer = input_layers[-1].clone()

            b, d, c, h, w = cur_input_layer.shape
            # self.try_to_log("before concatenate 2 slices", cur_input_layer.shape)
            cur_input_layer = cur_input_layer.reshape((b, d // 2, -1, h, w))
            # self.try_to_log("after  concatenate 2 slices", cur_input_layer.shape)

            self.try_to_info("layer_idx", layer_idx + 1, cur_input_layer.shape)

            num_slices = cur_input_layer.shape[1]
            for direction in range(self.num_directions):
                tmp = cur_input_layer.clone()
                cell_idx = layer_idx * self.num_directions + direction + self.num_directions
                hidden = hidden_state[cell_idx]
                convGRUCell = self.convGRUCells[cell_idx]
                op = -1 if direction == 1 else 1
                for t in range(num_slices)[::op]:
                    hidden = convGRUCell(input_tensor=tmp[:, t, :, :, :],
                                         cur_state=hidden)
                    cur_output_layer.append(hidden)
                    self.try_to_info(f"layer_idx {layer_idx + 1} - cell_idx {cell_idx}", f'slice ={t}', 'y, h, c',
                                     hidden.shape)

            if self.bidirectional:
                cur_output_layer_f = torch.stack(cur_output_layer[:num_slices], dim=1)
                cur_output_layer_b = torch.stack(cur_output_layer[-1:-num_slices - 1:-1], dim=1)
                self.try_to_info(f"layer_idx {layer_idx + 1} cur_output_layer forward and backward",
                                 cur_output_layer_f.shape, cur_output_layer_b.shape)
                cur_output_layer = cur_output_layer_f + cur_output_layer_b
            else:
                cur_output_layer = torch.stack(cur_output_layer, dim=1)

            self.try_to_info(f"layer_idx {layer_idx + 1} cur_output_layer", cur_output_layer.shape)
            input_layers.append(cur_output_layer)
            # last_state_list.append([hidden, cell])

        for i in range(len(input_layers)):
            # b, d, c, h, w => b, c, d, h, w
            input_layers[i] = input_layers[i].permute(0, 2, 1, 3, 4)
            # print(input_layers[i].shape)

        input_layers.pop(0)
        # return input_layers
        return [input_layers[i] for i in self.out_indices]  # , last_state_list


if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss()
    x = torch.rand((2, 1, 8, 64, 64))
    convlstm = ConvGRUEncoder(dim=3, in_channels=1, base_width=3, stages=2, out_indices=(0, 1, 2), bidirectional=True)
    # print(convlstm)
    # print(convlstm.modules)
    convlstm.set_log()

    layer_output_list = convlstm(x)
    print("")
    for i in range(len(layer_output_list)):
        print(layer_output_list[i].shape)
