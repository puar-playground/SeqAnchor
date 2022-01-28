from model_dp import Anchor_dp
from torch import optim
from data_utility import *
from plot_utility import rainbow_cmap
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(0)
import random
random.seed(0)
import time
import matplotlib.pyplot as plt
import sys
import os


def trim_head(net, anchor_index, data_oh):

    trim_point = np.zeros([len(data_oh)])
    max_values = np.zeros([len(data_oh)])
    for i in range(len(data_oh)):
        ASM_out = np.asarray(net.ASM_matrix(data_oh[i], anchor_index))[-1, :]
        max_values[i] = np.max(ASM_out)
        trim_point[i] = np.argmax(ASM_out)

    mask = max_values > np.max(max_values) * 0.7
    trim_point = np.multiply(trim_point, mask)
    return trim_point.astype(int)


def Anchor_train(net, train_data, anchor_i, lr=5e-4, gamma=5, n_plot=np.inf):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('start training anchor:', str(anchor_i))
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    start = time.time()
    max_iter = 100 * len(train_data)

    for ep, (seq, seq_i) in enumerate(train_data):
        print('\rEpoch: ' + str(ep), end='')
        sk, kt, distribution = net(seq, anchor_index=anchor_i, gamma=gamma)
        loss = - sk - 5e-4 * kt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot
        if ep % n_plot == 0:
            # print('\nplot_dp: ' + str(int(ep / n_plot)))
            plt.plot(distribution[0, :].detach().numpy())
            plt.savefig('./plot_log/anchor_' + str(anchor_i) + '_iter_' + str(int(ep / n_plot)) + '.png')
            plt.close()
            torch.save(net.state_dict(), './model/anchor_' + str(anchor_i) + '_iter_' + str(int(ep / n_plot)) + '.pt')

        if ep == max_iter:
            print('\ndone training at epoch: %i' % ep)
            break

    print('anchor %i done, time cost: %4.2f seconds' % (anchor_i, time.time() - start))

    return net


if __name__ == "__main__":

    dir = './plot_log'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    n_anchor = 60
    n_seq = 1000
    net = Anchor_dp(kernel_size=20, anchor_n=n_anchor, norm_ratio=1)
    Train_Seq = SeqDataset('./Dataset/zymo/zymo_0.fa', n=n_seq)

    # load net initialization
    # net_state_dict = torch.load('./model/finished_anchor.pt')
    # net.load_state_dict(net_state_dict)

    for a in range(n_anchor):
        net = Anchor_train(net, Train_Seq, anchor_i=a, n_plot=10000)
        torch.save(net.state_dict(), './model/finished_anchor_' + str(a) + '.pt')
        trim_point = trim_head(net, a, Train_Seq.seq_oh)
        Train_Seq.update(trim_point)
        print('%i sequences remain' % len(Train_Seq.seq_list))
        print('sequence max length: %i' % Train_Seq.max_l)

        if len(Train_Seq.seq_list) < 2:
            break













