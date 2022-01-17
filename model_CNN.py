import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ASM import ASM_fast, ASM_convert_fast

class Anchor(nn.Module):

    def __init__(self, kernel_size, anchor_n):
        super(Anchor, self).__init__()
        self.kernel_size = kernel_size
        self.anchor_n = anchor_n
        self.softmax = torch.nn.Softmax(dim=2)
        self.cnn = nn.Sequential(
            nn.Conv1d(4, anchor_n, kernel_size=kernel_size, padding=0),
            # nn.ReLU(),
            nn.LeakyReLU(1e-4),
        )

    def Z_scores(self, distribution):
        l = distribution.shape[0]
        x = torch.tensor(list(range(l)))
        mean = torch.sum(torch.multiply(distribution, x))
        x_diff = x - mean
        var = torch.sum(torch.multiply(distribution, torch.pow(x_diff, 2.0)))
        std = torch.pow(var, 0.5)
        zscores = x_diff / std
        # skews = torch.mean(torch.pow(zscores, 3.0))
        # kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return zscores

    def Z_scores_tensor(self, distributions):
        l = distributions.shape[2]
        x = torch.tensor(list(range(l)))
        x = x.repeat(distributions.shape[0], distributions.shape[1], 1)
        mean = torch.sum(torch.multiply(distributions, x), dim=2)
        mean = mean[:, :, None].repeat(1, 1, l)
        x_diff = x - mean
        var = torch.sum(torch.multiply(distributions, torch.pow(x_diff, 2.0)), dim=2)
        std = torch.pow(var, 0.5)
        std = std[:, :, None].repeat(1, 1, l)
        zscores = x_diff / std
        return zscores

    @staticmethod
    def Skewness(zscores):
        skews = torch.mean(torch.pow(zscores, 3.0))
        return skews

    @staticmethod
    def Skewness_tensor(zscores):
        skews = torch.mean(torch.pow(zscores, 3.0), dim=2)
        return skews

    @staticmethod
    def Kurtosis(zscores):
        kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        return kurtosis

    @staticmethod
    def Kurtosis_tensor(zscores):
        kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=2) - 3.0
        return kurtosis

    def anchor_info(self, data, anchor_index):
        channel = self.cnn(data)
        channel_training = channel[:, anchor_index, :]
        channel_training = channel_training[None, :]
        max_info = torch.max(channel_training, dim=2)
        return (max_info[0].detach(), max_info[1])

    def channel_out(self, data, gamma=1):
        channel = self.cnn(data)
        channel = F.normalize(channel, dim=2, p=1)
        distribution = self.softmax(channel / gamma)
        return distribution

    def forward(self, data, anchor_index, gamma=1):
        channel = self.cnn(data)
        channel_training = channel[:, anchor_index, :]
        channel_training = channel_training[None, :]
        # print(channel_training.shape)
        channel = F.normalize(channel_training, dim=2, p=1)
        # distribution = self.softmax(channel / 1)
        distribution = self.softmax(channel / gamma)
        # print(distribution.shape)
        zscores = self.Z_scores_tensor(distribution)
        skews = self.Skewness_tensor(zscores)
        kts = self.Kurtosis_tensor(zscores)

        return skews, kts, channel



if __name__ == "__main__":

    net = Anchor(kernel_size=20, anchor_n=10)
    data = torch.randn(1, 4, 1000)

    skews, kts, distribution = net(data, anchor_index=0)
    # print(skews)
    # print(kts)
    #
    # max_info = net.anchor_info(data, 0)
    # print(max_info)






