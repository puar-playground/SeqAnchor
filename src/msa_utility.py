from test_score import aop
from data_utility import *
import torch
torch.manual_seed(0)
import random
random.seed(0)
import time
import pandas as pd
from ASM import ASM_fast, anchor_align_fast, nw_align, asm_align, nw_profile_align, asm_profile_align
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%5.2f" % x))


def alignment_to_profile(seqs):
    basis = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '-': -1}
    l = max([len(s) if s is not None else 0 for s in seqs])
    profile = np.zeros([4, l])
    for s in seqs:
        if s is not None and len(s) != 0:
            x = [basis[c] for c in s]
            oh = np.concatenate((np.eye(4), -0.5*np.ones((4, 1))), axis=1)[:, x]
            profile += oh
    profile *= 1/len(seqs)
    # profile = profile / np.linalg.norm(profile, axis=0)
    return profile


def pattern_gaps(p_align, p_length):
    gap_count = np.zeros([p_length + 1]).astype(int)
    ind = 0
    for c in p_align:
        if c == '*':
            ind += 1
        else:
            gap_count[ind] += 1
    return gap_count


def starline_expend(s, starline):
    s_expend = ''
    ind = 0
    for c in starline:
        if c == '-':
            s_expend += '-'
        else:
            s_expend += s[ind]
            ind += 1
    return s_expend


def merge_anchor(anchor_align_list, p_length):

    merged_output = []
    gap_count_all = np.zeros([1000, p_length + 1]).astype(int)
    bas_log = []
    n_seq = len(anchor_align_list)
    for i in range(n_seq):
        if anchor_align_list[i][1] != '':
            gap_count_all[i, :] = pattern_gaps(anchor_align_list[i][0], p_length)
        bas_log.append(anchor_align_list[i][1])

    gap_cumulate = np.max(gap_count_all, axis=0).astype(int)
    gap_count_all = gap_count_all.astype(int)

    for i, s_align in enumerate(bas_log):
        if s_align == '':
            merged_output.append('')
            continue
        gap_count = gap_count_all[i, :]
        gap_insert = gap_cumulate - gap_count
        s_merge = []
        p_index = 0
        for insert_i, insert_n in enumerate(gap_insert):
            insert_str = ''
            # insertion added
            if insert_n > 0:
                insert_str += ''.join(['-' for _ in range(insert_n)])
            # insertion own
            if gap_count[insert_i] > 0:
                insert_str += s_align[p_index:p_index + gap_count[insert_i]]
                p_index += gap_count[insert_i]
            s_merge.append(insert_str)

            # match
            if p_index < len(s_align):
                match_str = s_align[p_index]
                s_merge.append(match_str)
                p_index += 1
        #
        # print(''.join(s_merge))
        merged_output.append(''.join(s_merge))

    return merged_output


def anchoring(net, Train_Seq, th=0.6):
    data_oh = Train_Seq.seq_oh
    aligned_anchor_array = [[] for _ in range(len(data_oh))]
    anchor_position = [[] for _ in range(len(data_oh))]
    trim_point = np.zeros([len(data_oh), net.anchor_n])

    for anchor_index in range(net.anchor_n):
        anchor_align_array = []
        max_values = np.zeros([len(data_oh)])
        trim_temp = np.zeros([len(data_oh)])
        head = trim_point[:, max(0, anchor_index - 1)].astype(int)
        for i in range(len(data_oh)):

            data_oh_trim = data_oh[i][0, :, head[i]:].numpy().astype(np.float64)

            if data_oh_trim.shape[1] < 20:
                anchor_align_array.append(('', ''))
                anchor_position[i].append((None, None))
                trim_temp[i] = 0
            else:
                p = net.patterns.detach().numpy()[anchor_index, :, :].astype(np.float64)
                p_align, s_align, (j_min, j_max), v = anchor_align_fast(p, data_oh_trim, 1)
                anchor_position[i].append((head[i]+j_min, head[i]+j_max))
                anchor_align_array.append((p_align, s_align))
                max_values[i] = v
                trim_temp[i] = j_max

        keep_mask = (max_values > th * np.max(max_values))
        anchor_align_array = [x if keep else ('', '') for keep, x in zip(keep_mask, anchor_align_array)]
        anchor_position = [x if keep else x[:-1] + [(None, None)] for keep, x in zip(keep_mask, anchor_position)]
        trim_point[:, anchor_index] = np.multiply(trim_temp, keep_mask) + head
        merged_output = merge_anchor(anchor_align_array, net.kernel_size)

        # aligned_anchor_pd[anchor_index] = merged_output
        aligned_anchor_array = [x + [s] for (x, s) in zip(aligned_anchor_array, merged_output)]
    return aligned_anchor_array, anchor_position


def inter_retriving(anchor_position, Train_Seq):
    n_anchor = max([len(x) for x in anchor_position])
    n_seq = len(anchor_position)
    # intervals_pd = pd.DataFrame(columns=range(n_anchor + 1), index=range(n_seq))
    intervals = [[None for _ in range(n_anchor + 1)] for _ in range(n_seq)]
    shortcuts = [[] for _ in range(n_seq)]

    for i in range(n_seq):
        s = Train_Seq.seq_list[i]
        pos_list = [(0, 0)] + anchor_position[i] + [(len(s), len(s))]
        for a in range(len(pos_list) - 1):
            left = pos_list[a][1]
            right = pos_list[a + 1][0]
            if (left is not None) and (right is not None):
                intervals[i][a] = s[left:right]
                # intervals_pd[a][i] = s[left:right]
            else:
                if (left is not None) and (right is None):
                    left_st = left
                    a_left = a
                elif (left is None) and (right is not None):
                    right_st = right
                    a_right = a
                    # from a_left -th interval to a_right -th interval (end points included)
                    # (2, 3): aligned_intervals[i][2] + aligned_anchors[i][2] + aligned_intervals[i][3]
                    shortcuts[i].append((s[left_st:right_st], str(a_left) + '-' + str(a_right)))

    return intervals, shortcuts


def inter_aligning(intervals, n_anchor, mode='asm'):
    align_function = {'asm': asm_align, 'nw': nw_align}
    n_seq = len(intervals)
    n_inter = n_anchor + 1
    # intervals_pd = pd.DataFrame(columns=range(n_inter), index=range(n_seq))
    aligned_intervals = [[None for _ in range(n_inter)] for _ in range(n_seq)]
    for i in range(n_inter):
        same_inters = sorted(set([x[i] for x in intervals if x[i] and x[i] != '']), key=lambda x: -len(x))
        if len(same_inters) == 0:
            for s in range(n_seq):
                if intervals[s][i] is not None:
                    aligned_intervals[s][i] = ''
        else:
            inter_align_array = []
            represent = same_inters[0]
            for s in same_inters:
                # p_align, s_align = align_function[mode](represent, s, 1)
                s_align, p_align = align_function[mode](s, represent, 1)
                inter_align_array.append((p_align, s_align))

            merged_inters = merge_anchor(inter_align_array, len(represent))
            inter_dict = {k: v for k, v in zip(same_inters, merged_inters)}
            inter_dict[''] = ''.join(['-' for _ in range(len(merged_inters[0]))])
            for s in range(n_seq):
                if intervals[s][i] is not None:
                    aligned_intervals[s][i] = inter_dict[intervals[s][i]]

    return aligned_intervals


def representative_msa(seq_list, mode='asm'):
    align_function = {'asm': asm_align, 'nw': nw_align}
    same_seq = sorted(set(seq_list), key=lambda x: -len(x))
    inter_align_array = []
    represent = same_seq[0]
    for s in same_seq:
        s_align, p_align = align_function[mode](s, represent, 1)
        inter_align_array.append((p_align, s_align))
    merged_align = merge_anchor(inter_align_array, len(represent))
    shortcut_dict = {k: v for k, v in zip(same_seq, merged_align)}
    return shortcut_dict


def shortcut_adapting(shortcuts, anchor_profile, inter_profile, mode='asm'):
    # initialize gap array for representative msa gap accumulation
    anchor_gap_array = [np.zeros([ap.shape[1] - 1]).astype(int) for ap in anchor_profile]
    inter_gap_array = [np.zeros([ip.shape[1] + 1]).astype(int) for ip in inter_profile]

    # a dict using all skip (from which inter to which inter) as keys, all shortcut strings as values for the skip
    sc = dict()
    for i in range(n_seq):
        if len(shortcuts[i]) != 0:
            for x in shortcuts[i]:
                if x[1] not in sc.keys():
                    sc[x[1]] = [x[0]]
                else:
                    sc[x[1]].append(x[0])

    # get profile for each skip type and gap index array
    scp_align_results = dict()
    sc_align_lookup = dict()

    for k in sc.keys():
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(k)
        # get all the shortcuts of the same skip
        sc_unique = sorted(set(sc[k]), key=lambda x: -len(x))
        out_dict = representative_msa(sc_unique, mode=mode)
        sc_align_lookup[k] = out_dict
        sc_list = [out_dict[sc] for sc in sc[k]]
        # print(sc_list)
        # compute the alignment profile of the shortcut msa
        shortcut_profile = alignment_to_profile(sc_list)
        # concatenate the ref profile and initialize gap index array
        on_sc_gap_index = []
        bounds = [int(x) for x in k.split('-')]
        ref_profile = np.zeros([4, 0]).astype(int)
        for b in range(bounds[0], bounds[1]):
            on_sc_gap_index += [('inter', b, _) for _ in range(inter_profile[b].shape[1] + 1)] + [('anchor', b, _) for _ in range(anchor_profile[b].shape[1] - 1)]
            ref_profile = np.concatenate((ref_profile, inter_profile[b], anchor_profile[b]), axis=1)
        ref_profile = np.concatenate((ref_profile, inter_profile[bounds[1]]), axis=1)
        on_sc_gap_index += [('inter', bounds[1], _) for _ in range(inter_profile[bounds[1]].shape[1] + 1)]

        if ref_profile.shape[1] >= shortcut_profile.shape[1]:
            scp_aligned, refp_aligned = asm_profile_align(shortcut_profile, ref_profile, 1)
        else:
            refp_aligned, scp_aligned = asm_profile_align(ref_profile, shortcut_profile, 1)

        scp_align_results[k] = (scp_aligned, refp_aligned)

        ind = 0
        for c in refp_aligned:
            if c == '*':
                ind += 1
            else:
                # print('gap index on ref profile: %i' % ind)
                ind_info = on_sc_gap_index[ind]
                # print(ind_info)
                if ind_info[0] == 'inter':
                    inter_gap_array[ind_info[1]][ind_info[2]] += 1
                else:
                    anchor_gap_array[ind_info[1]][ind_info[2]] += 1

    for k in sc_align_lookup.keys():
        # print(k)
        # print(scp_align_results[k])
        # print(sc_align_lookup[k])
        bounds = [int(x) for x in k.split('-')]
        ref_gap_array = np.zeros([0]).astype(int)
        for b in range(bounds[0], bounds[1]):
            ref_gap_array = np.concatenate((ref_gap_array, inter_gap_array[b], anchor_gap_array[b]))
        ref_gap_array = np.concatenate((ref_gap_array, inter_gap_array[bounds[1]]))
        sc_gap_array = pattern_gaps(scp_align_results[k][1], ref_gap_array.shape[0] - 1)
        gap_insert = ref_gap_array - sc_gap_array
        if np.sum(gap_insert) > 0:
            s_merge = []
            p_index = 0
            for insert_i, insert_n in enumerate(gap_insert):
                insert_str = ''
                # insertion added
                if insert_n > 0:
                    insert_str += ''.join(['-' for _ in range(insert_n)])
                # insertion own
                if sc_gap_array[insert_i] > 0:
                    insert_str += scp_align_results[k][0][p_index:p_index + sc_gap_array[insert_i]]
                    p_index += sc_gap_array[insert_i]
                s_merge.append(insert_str)

                # match
                if p_index < len(scp_align_results[k][0]):
                    match_str = scp_align_results[k][0][p_index]
                    s_merge.append(match_str)
                    p_index += 1
            starline = ''.join(s_merge)
        else:
            starline = scp_align_results[k][0]
        # print('starline', starline)

        for sc_k in sc_align_lookup[k].keys():
            sc_merged = sc_align_lookup[k][sc_k]
            sc_align_lookup[k][sc_k] = starline_expend(sc_merged, starline)

    # use sc_align_lookup to align all shortcuts and output
    for i in range(n_seq):
        if len(shortcuts[i]) != 0:
            # print(shortcuts[i])
            shortcuts[i] = [(sc_align_lookup[pair_ind][sc], pair_ind) for (sc, pair_ind) in shortcuts[i]]
            # print(shortcuts[i])

    return anchor_gap_array, inter_gap_array, shortcuts


def anchors_adapting(aligned_anchors_pd, anchor_gap_array):
    for i, gap_array in enumerate(anchor_gap_array):
        if np.sum(gap_array) != 0:
            for g in range(len(gap_array)):
                if gap_array[g] > 0:
                    insert_g = ''.join(['-' for _ in range(gap_array[g])])
                    aligned_anchors_pd[i] = [s[:g+1] + insert_g + s[g+1:] if s is not None else None for s in
                                               aligned_anchors_pd[i]]
    return aligned_anchors_pd


def inters_adapting(aligned_intervals_pd, inter_gap_array):
    for i, gap_array in enumerate(inter_gap_array):
        if np.sum(gap_array) != 0:
            for g in range(len(gap_array)):
                if gap_array[g] > 0:
                    insert_g = ''.join(['-' for _ in range(gap_array[g])])
                    aligned_intervals_pd[i] = [s[:g] + insert_g + s[g:] if s is not None else None for s in
                                               aligned_intervals_pd[i]]
    return aligned_intervals_pd


def partial_assemble(aligned_anchors_pd, aligned_intervals_pd, i, start, end):
    merged_s = ''
    for a in range(start, end):
        merged_s += aligned_anchors_pd[a][i] + aligned_intervals_pd[a+1][i]
    merged_s += aligned_anchors_pd[end][i]
    return merged_s


def msa_assemble(aligned_anchors_pd, aligned_intervals_pd, shortcuts_aligned):
    n_seq = len(shortcuts_aligned)
    n_anchor = aligned_anchors_pd.shape[1]
    msa = ['' for _ in range(n_seq)]
    for i in range(n_seq):
        if len(shortcuts_aligned[i]) == 0:
            msa[i] = aligned_intervals_pd[0][i] + \
                     partial_assemble(aligned_anchors_pd, aligned_intervals_pd, i, 0, n_anchor - 1) \
                     + aligned_intervals_pd[n_anchor][i]
        else:
            start = 0
            for sc in shortcuts_aligned[i]:
                end = int(sc[1].split('-')[0]) - 1
                if start == 0 and end >= 0:
                    msa[i] += aligned_intervals_pd[start][i] + \
                              partial_assemble(aligned_anchors_pd, aligned_intervals_pd, i, start, end) + sc[0]
                elif start > 0 and end >= 0:
                    msa[i] += partial_assemble(aligned_anchors_pd, aligned_intervals_pd, i, start, end) + sc[0]
                else:
                    msa[i] += sc[0]
                start = int(sc[1].split('-')[1])

            if start < n_anchor - 1:
                msa[i] += partial_assemble(aligned_anchors_pd, aligned_intervals_pd, i, start, n_anchor - 1) + \
                          aligned_intervals_pd[n_anchor][i]

    return msa


def msa_compress_gap(msa):
    accumulate = np.zeros([len(msa[0])]).astype(int)
    for s in msa:
        accumulate += np.array([1 if c == '-' else 0 for c in s]).astype(int)
    remove_ind = set(np.where(accumulate == len(msa))[0])
    msa_compressed = [''.join([c for idx, c in enumerate(s) if idx not in remove_ind]) for s in msa]
    return msa_compressed


if __name__ == "__main__":

    start = time.time()
    n_anchor = 45
    n_seq = 1000
    net = Anchor_dp(20, anchor_n=n_anchor, norm_ratio=1)
    net_state_dict = torch.load('../ckpt/finished_anchors.pt')
    net.load_state_dict(net_state_dict)

    Train_Seq = SeqDataset('./Dataset/zymo/zymo_0.fa', n=n_seq)
    print('Data loaded, cost: %4.2fs' % (time.time() - start))

    start = time.time()
    # anchoring
    aligned_anchors, anchor_position = anchoring(net, Train_Seq, th=0.6)
    aligned_anchors_pd = pd.DataFrame(data=aligned_anchors, columns=range(n_anchor), index=range(n_seq))
    # get intervals and do representative msa
    intervals, shortcuts = inter_retriving(anchor_position, Train_Seq)
    inter_pd = pd.DataFrame(data=intervals, columns=range(n_anchor + 1), index=range(n_seq))
    aligned_intervals = inter_aligning(intervals, n_anchor, mode='asm')
    aligned_intervals_pd = pd.DataFrame(data=aligned_intervals, columns=range(n_anchor + 1), index=range(n_seq))

    # get profiles
    anchor_profile = [alignment_to_profile(aligned_anchors_pd[i]) for i in range(n_anchor)]
    inter_profile = [alignment_to_profile(aligned_intervals_pd[i]) for i in range(n_anchor + 1)]

    # process shortcut
    anchor_gap_array, inter_gap_array, shortcuts_aligned = shortcut_adapting(shortcuts, anchor_profile,
                                                                             inter_profile, mode='nw')
    aligned_anchors_pd = anchors_adapting(aligned_anchors_pd, anchor_gap_array)
    aligned_intervals_pd = inters_adapting(aligned_intervals_pd, inter_gap_array)

    # msa merge
    msa = msa_assemble(aligned_anchors_pd, aligned_intervals_pd, shortcuts_aligned)
    print('MSA done cost: %4.2fs' % (time.time() - start))

    msa = msa_compress_gap(msa[:100])
    score = aop(msa)
    print('AOP score of the first 100 sequences: %i' % score)

    for s in msa:
        print(s)


