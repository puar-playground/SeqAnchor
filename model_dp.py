import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ASM import ASM_fast, ASM_convert_fast

class Anchor_dp(nn.Module):
    def __init__(self, kernel_size, anchor_n, norm_ratio=1):
        super(Anchor_dp, self).__init__()
        self.norm_ratio = norm_ratio
        self.kernel_size = kernel_size
        self.anchor_n = anchor_n
        self.softmax = torch.nn.Softmax(dim=1)
        self.in_dim = 4
        self.activation = nn.LeakyReLU(1e-3)
        self.patterns = torch.nn.Parameter(torch.rand([anchor_n, self.in_dim, kernel_size]))
        self.bias = torch.nn.Parameter(- 0.4 * kernel_size * torch.ones([anchor_n]))

    def state_dict(self):
        state = dict()
        state['patterns'] = self.patterns
        state['bias'] = self.bias
        return state

    # def load_state_dict(self, state):
    #     n, d, l = torch.nn.Parameter(state['patterns']).shape
    #     if not (self.anchor_n==n and self.in_dim==d and self.kernel_size==l):
    #         raise("weight dimension not compatible")
    #
    #     self.patterns = torch.nn.Parameter(state['patterns'])
    #     self.bias = torch.nn.Parameter(state['bias'])
    #     self.pattern_normlize()

    def load_state_dict(self, state):
        n, d, l = torch.nn.Parameter(state['patterns']).shape
        if not (self.anchor_n <= n and self.in_dim == d and self.kernel_size==l):
            raise("weight dimension not compatible")

        self.patterns = torch.nn.Parameter(state['patterns'][:self.anchor_n, :, :])
        self.bias = torch.nn.Parameter(state['bias'][:self.anchor_n])
        self.pattern_normlize()

    def Z_scores(self, distribution):
        l = distribution.shape[1]
        x = torch.tensor(list(range(l)))
        mean = torch.sum(torch.multiply(distribution, x))
        x_diff = x - mean
        var = torch.sum(torch.multiply(distribution, torch.pow(x_diff, 2.0)))
        std = torch.pow(var, 0.5)
        zscores = x_diff / std
        # skews = torch.mean(torch.pow(zscores, 3.0))
        # kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return zscores

    @staticmethod
    def Skewness(zscores):
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    @staticmethod
    def Kurtosis(zscores):
        kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return kurtosis

    def ASM_matrix(self, data, anchor_index):
        self.pattern_normlize()
        np_pattern = self.patterns[anchor_index, :, :].detach().numpy().astype(np.float64)
        np_data = data.numpy().astype(np.float64)[0, :, :]
        F_fast = ASM_fast(np_pattern, np_data, 1)
        return F_fast

    def channel_array_convert(self, data, pattern):
        np_pattern = pattern.detach().numpy().astype(np.float64)
        np_data = data.numpy().astype(np.float64)[0, :, :]
        F_fast = ASM_fast(np_pattern, np_data, 1)
        matched_array, gap_array = ASM_convert_fast(F_fast, np_pattern, np_data, 1)
        matched_array = torch.tensor(matched_array)
        gap_array = torch.tensor(gap_array)
        product = torch.sum(torch.multiply(pattern, matched_array), dim=[1, 2])
        ASM_out = product - gap_array
        return ASM_out

    def forward(self, data, anchor_index, gamma=1):
        pattern = self.norm_ratio * F.normalize(self.patterns[anchor_index, :, :], p=2, dim=0)
        ASM_out = self.channel_array_convert(data, pattern)
        channel = self.activation(ASM_out + self.bias[anchor_index])
        channel = channel[None, :]
        channel_norm = F.normalize(channel, p=1)
        # distribution = self.softmax(channel / 1)
        distribution = self.softmax(channel_norm / gamma)
        # print(distribution.shape)
        zscores = self.Z_scores(distribution)
        skews = self.Skewness(zscores)
        kts = self.Kurtosis(zscores)

        return skews, kts, channel

    def pattern_normlize(self):
        self.patterns = nn.Parameter(self.norm_ratio * F.normalize(self.patterns, p=2, dim=1))


if __name__ == "__main__":

    net = Anchor_dp(kernel_size=20, anchor_n=10)
    data = torch.abs(torch.randn(1, 4, 1000))
    data = torch.floor(torch.divide(data, torch.max(data, dim=1)[0]))

    skews, kts, distribution = net(data, anchor_index=0)

    print(skews, kts)

    ASM_matrix = net.ASM_matrix(data, anchor_index=0)
    print(list(np.asarray(ASM_matrix)[-1, 20:]))


