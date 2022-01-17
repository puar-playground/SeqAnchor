import random
random.seed(0)
from model_dp import Anchor_dp
import numpy as np
np.random.seed(0)
import torch
from torch.utils.data import Dataset

def one_hot(s):

    basis = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    l = len(s)
    feature = torch.zeros([4, l])
    for i, c in enumerate(s):
        if c not in ['A', 'T', 'G', 'C']:
            continue
        else:
            feature[basis[c], i] = 1
    return feature


class SeqDataset(Dataset):

    def __init__(self, seq_dir, n):
        self.n = n
        self.seq_list = self.read_seq(seq_dir)
        self.max_l = max([len(s) for s in self.seq_list])
        self.seq_oh = self.one_hot_list()

    def __getitem__(self, index):
        # (seq, ind) = self.seq_rotation(index)
        ind = random.randint(0, min([self.n - 1, len(self.n) - 1]))
        seq = self.seq_oh[ind]
        return (seq, ind)

    def __len__(self):
        return self.n

    def read_seq(self, seq_dir):
        f_s = open(seq_dir, 'r')
        seq_list = []
        while 1:
            name = f_s.readline()[1:-1]
            if not name or len(seq_list) >= self.n:
                break
            seq = f_s.readline()[:-1]
            seq_list.append(seq)
        return seq_list

    def one_hot_list(self):
        basis = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        feature_list = []
        for j, s in enumerate(self.seq_list):
            feature = torch.zeros([1, 4, len(s)])
            for i, c in enumerate(s):
                if c not in ['A', 'T', 'G', 'C']:
                    continue
                else:
                    feature[0, basis[c], i] = 1
            feature_list.append(feature)
        return feature_list

    def seq_rotation(self, index):
        index_valid = index % self.n
        return (self.seq_oh[index_valid], index_valid)


    def update(self, trim_point):
        self.seq_list = [s[ind:] for (s, ind) in zip(self.seq_list, trim_point) if len(s[ind:]) > 20]
        self.seq_oh = self.one_hot_list()
        if len(self.seq_list) != 0:
            self.max_l = max([len(s) for s in self.seq_list])
        else:
            self.max_l = 0


if __name__=='__main__':

    SeqData = SeqDataset('/Users/chenjian/Desktop/ASM/Seq_Data/Greengenes/1000_pairs/train/gg_0.fa', 1000)

    for i, seq in enumerate(SeqData):
        print(i, seq)
        break

    net = Anchor_dp(20, anchor_n=1)
    net_state_dict = torch.load('./model/anchor_0_done.pt')
    net.load_state_dict(net_state_dict)







