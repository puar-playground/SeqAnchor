from model_CNN import *
from torch import optim
from data_utility import *
from plot_utility import rainbow_cmap
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(0)
import time
import matplotlib.pyplot as plt
import sys
import os


class Anchor_log(object):
    def __init__(self, n_anchor, n_seq):
        self.log = np.inf * np.ones([n_seq, n_anchor])
        self.n_seq = n_seq
        self.n_anchor = n_anchor

    def update(self, net, train_data, anchor_i):
        print('=================================================================================>')
        print('update log')
        for i in range(self.n_seq):
            if anchor_i > 0:
                loc_head = int(self.log[i, anchor_i - 1])
                seq = train_data.seq_oh[i][:, :, loc_head + net.kernel_size:]
                max_info = net.anchor_info(seq, anchor_i)
                self.log[i, anchor_i] = max_info[1].item() + loc_head + net.kernel_size
            else:
                seq = train_data.seq_oh[i]
                max_info = net.anchor_info(seq, anchor_i)
                self.log[i, anchor_i] = max_info[1].item()


def Anchor_train(net, train_data, anchor_i, log_file, lr=1e-4, gamma=1, n_plot=10):

    print('start training anchor:', str(anchor_i))
    min_loss_avg = 100
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    start = time.time()
    max_iter = 1000 * len(train_data)
    plt_boolean = 0
    plt_cnt = 0
    cmap = rainbow_cmap(20)
    loss_buffer = 100 * np.ones(len(train_data))
    for ep, (seq, seq_i) in enumerate(train_data):
        if anchor_i > 0:
            loc_head = int(log_file[seq_i, anchor_i - 1])
            seq_train = seq[:, :, loc_head + net.kernel_size:]
        else:
            seq_train = seq

        sk, kt, distribution = net(seq_train, anchor_index=anchor_i, gamma=gamma)
        loss = - sk[0, 0] - 1e-4 * (kt[0, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_buffer[seq_i] = loss.item()
        loss_avg = np.mean(loss_buffer)

        if loss_avg < min_loss_avg:
            min_loss_avg = loss_avg
            patience = 0
        else:
            patience += 1
            if patience == len(train_data) * 50:
                break
            # if not plt_boolean:
            #     continue

        # plot
        if ep % (len(train_data) * n_plot) == 0:
            plt_boolean = 1
            plt_cnt = 0

        if plt_boolean:
            if plt_cnt == 0:
                print('=================================================================================>')
                print('plot: ' + str(int(ep / (len(train_data) * n_plot))))
                fig, axes = plt.subplots(10, 1, figsize=(20, 10))


            print('epoch %i' % ep)
            distribution_plot = net.channel_out(seq, gamma=gamma)
            fig.suptitle('Sequences')

            for a in range(anchor_i + 1):
                channel_plot = distribution_plot[0, a, :].detach().numpy()
                if a > 0:
                    loc_head = int(log_file[seq_i, a - 1])
                    channel_plot[:loc_head + net.kernel_size] = None
                if a != anchor_i:
                    max_ind = np.nanargmax(channel_plot)
                    channel_plot[max_ind:max_ind + net.kernel_size] = channel_plot[max_ind]
                    axes[plt_cnt].plot(channel_plot, color=cmap(a))
                else:
                    axes_learn = axes[plt_cnt].twinx()
                    axes_learn.plot(channel_plot, color=cmap(a))

            plt_cnt += 1

            if plt_cnt == 10:
                plt_boolean = 0
                plt_cnt = 0
                plt.savefig('./plot/anchor_' + str(anchor_i) + '_iter_' + str(int(ep / (len(train_data) * n_plot))) + '.png')
                plt.close()

            if ep == max_iter:
                break

    print('anchor %i done, time cost: %4.2f seconds' % (anchor_i, time.time() - start))

    return net


if __name__ == "__main__":

    dir = './plot'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    n_anchor = 1
    n_seq = 20
    net = Anchor(20, anchor_n=n_anchor)
    Train_Seq = SeqDataset('/Users/chenjian/Desktop/ASM/Seq_Data/Greengenes/1000_pairs/train/gg_0.fa', n=n_seq)
    Anchor_history = Anchor_log(n_anchor, n_seq)

    for a in range(n_anchor):
        net = Anchor_train(net, Train_Seq, anchor_i=a, log_file=Anchor_history.log)
        Anchor_history.update(net, Train_Seq, a)

    print(Anchor_history.log)










