"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

Main A3C model which outputs predicted value, action logits and hidden state.
Includes helper functions too for weight initialisation and dynamically computing LSTM/flatten input
size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    """
    Weights are normalized over their column. Also, allows control over std which is useful for
    initialising action logit output so that all actions have similar likelihood
    """
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class FrameEncoder(torch.nn.Module):
    def __init__(self, num_input_channels, frame_dim):
        super(FrameEncoder, self).__init__()
        self.frame_dim = frame_dim
        self.num_filter = 8
        self.stride = 2
        self.kernel_size = 3
        self.padding = 1
        self.conv1 = nn.Conv2d(num_input_channels, 64, self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(64, 32, self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv3 = nn.Conv2d(32, 16, self.kernel_size, stride=self.stride, padding=self.padding)
        self.conv4 = nn.Conv2d(16, self.num_filter, self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        return x

    def calculate_lstm_input_size(self):
        """
        Assumes square resolution image. Find LSTM size after 4 conv layers below in A3C using regular
        Convolution math. For example:
        42x42 -> (42 − 3 + 2)÷ 2 + 1 = 21x21 after 1 layer
        11x11 after 2 layers -> 6x6 after 3 -> and finally 3x3 after 4 layers
        Therefore lstm input size after flattening would be (3 * 3 * num_filters)
        """
        width = (self.frame_dim - self.kernel_size + 2 * self.padding) // self.stride + 1
        width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        return width * width * self.num_filter

class PointCloudEncoder(torch.nn.Module):
    def __init__(self):
        super(PointCloudEncoder, self).__init__()
        self.n_agent_feature = 5
        self.ori_point_dim = 3
        self.ori_feature_dim = 1
        self.n_point_feature = 64
        self.shared_mlp = nn.Conv1d(in_channels=self.ori_point_dim, out_channels=self.n_point_feature,
                                    kernel_size=1, stride=1)

    def forward(self, inputs):
        obj_point_and_feature, agent_feature = inputs
        obj_point, obj_feature = obj_point_and_feature[:,:self.ori_point_dim,:], \
                                 obj_point_and_feature[:,self.ori_point_dim:self.ori_point_dim + self.ori_feature_dim, :]

        point_features = self.shared_mlp(obj_point)
        aggregate_features = torch.matmul(obj_feature, point_features.transpose(1,2)).flatten(start_dim=1)
        return torch.cat([aggregate_features, agent_feature], dim=1)

    def calculate_lstm_input_size(self):
        return self.n_point_feature + self.n_agent_feature




class ActorCritic(torch.nn.Module):
    """
    Mainly Ikostrikov's implementation of A3C (https://arxiv.org/abs/1602.01783).

    Processes an input image (with num_input_channels) with 4 conv layers,
    interspersed with 4 elu activation functions. The output of the final layer is then flattened
    and passed to an LSTM (with previous or initial hidden and cell states (hx and cx)).
    The new hidden state is used as an input to the critic and value nn.Linear layer heads,
    The final output is then predicted value, action logits, hx and cx.
    """

    def __init__(self, num_outputs, num_input_channels=None, frame_dim=None):
        super(ActorCritic, self).__init__()

        if num_input_channels is not None and frame_dim is not None:
            self.feature_encoder = FrameEncoder(num_input_channels, frame_dim)
        else:
            assert num_input_channels is None and frame_dim is None
            self.feature_encoder = PointCloudEncoder()

        self.lstm_cell_size = self.feature_encoder.calculate_lstm_input_size()
        self.lstm = nn.LSTMCell(self.lstm_cell_size, 256)  # for 128x128 input

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.feature_encoder(inputs)
        x = x.view(-1, self.lstm_cell_size)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
