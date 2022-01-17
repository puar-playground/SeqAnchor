import numpy as np
import torch
import matplotlib.pyplot as plt

def skewness(distribution, gamma):
    l = distribution.shape[0]
    x = torch.tensor(list(range(l)))
    softmax = torch.nn.Softmax(dim=0)
    distribution = softmax(torch.tensor(distribution) / gamma)
    # distribution = torch.tensor(distribution) / np.sum(distribution)
    mean = torch.sum(torch.multiply(distribution, x))
    x_diff = x - mean
    var = torch.sum(torch.multiply(distribution, torch.pow(x_diff, 2.0)))
    std = torch.pow(var, 0.5)
    zscores = x_diff / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    return skews


if __name__ == '__main__':

    l = 20
    gamma = 1
    s_list = []
    softmax = torch.nn.Softmax(dim=0)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    for peak in range(l):
        distribution = 0. * np.random.random([l])
        distribution[peak] = 10
        distribution = distribution / max(distribution)

        s = skewness(distribution, gamma)
        s_list.append(s)

        plt.plot(distribution)

    plt.subplot(1, 2, 2)
    plt.plot(s_list)
    plt.xticks(list(range(l)))
    plt.xlabel('spike_position')
    plt.ylabel('skewness')
    plt.savefig('skewness_show.png')


