import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()
        self.shape = (input_shape[2], input_shape[0], input_shape[1])
        act = nn.ELU
        depth, kernels = info['depth'], info['kernels']
        layers = []
        for i, kernel in enumerate(kernels):
            if i == 0:
                inp_dim = self.shape[0]
            else:
                inp_dim = 2 ** (i - 1) * depth
            cur_depth = 2 ** i * depth
            layers.append(nn.Conv2d(inp_dim, cur_depth, kernel, 2))
            layers.append(act())
        self.layers = nn.Sequential(*layers)
        # """there is no specific embed_size in official code?"""
        with torch.no_grad():
            test_x = torch.randn(self.shape).unsqueeze(0)
            self.conv_out_shape = self.layers(test_x).shape[1:]  # save for obs_decoder_init
        conv_out_size = np.prod(self.conv_out_shape).item()
        if embedding_size == conv_out_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(conv_out_size, embedding_size)

    def forward(self, obs):
        """permute added: (~, h, w, c) -> (~, c, h, w)"""
        num_dims = len(obs.shape)
        new_order = tuple(range(num_dims))[:-3] + (num_dims - 1, num_dims - 3, num_dims - 2)
        obs = obs.permute(*new_order)

        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed


class ObsDecoder(nn.Module):
    def __init__(self, output_shape, modelstate_size, convt_in_shape, info):
        """
        :param output_shape: tuple containing shape of output obs (H, W, C)
        :param embed_size: the size of input vector, for dreamerv2 : modelstate
        :param convt_in: the size of input vector of convt
        """
        super(ObsDecoder, self).__init__()
        """save the shape of (h, w, c) for foward"""
        self.output_shape = output_shape
        output_shape = (output_shape[2], output_shape[0], output_shape[1])
        depth, kernels = info['depth'], info['kernels']
        # self.convt_in_shape = (8 * depth, 2, 2)  # (384, 2, 2)
        self.convt_in_shape = convt_in_shape
        inp_dim = np.prod(self.convt_in_shape).item()
        if modelstate_size == inp_dim:  # inp_dim: 1536
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(modelstate_size, inp_dim)
        layers = []
        for i, kernel in enumerate(kernels):
            cur_depth = 2 ** (len(kernels) - i - 2) * depth
            act = nn.ELU
            if i == len(kernels) - 1:
                cur_depth = output_shape[0]
                act = None
            if i != 0:
                inp_dim = 2 ** (len(kernels) - (i - 1) - 2) * depth
            layers.append(nn.ConvTranspose2d(inp_dim, cur_depth, kernel, 2))
            if act is not None:
                layers.append(act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.convt_in_shape))
        x = self.decoder(x)  # (3136, 3, 56, 56)
        """permute"""
        num_dims = len(x.shape)
        new_order = tuple(range(num_dims))[:-3] + (num_dims - 2, num_dims - 1, num_dims - 3)
        x = x.permute(*new_order)

        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
