#cython: boundscheck=False, cdivision=True, wraparound=False
import numpy as np
from libc.stdio cimport printf

def ASM_fast(double[:, :] seq1, double[:, :] seq2, int gap):
    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L1 + 1, L2 + 1])
    cdef int i, j, d
    cdef double p_max = 0

    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        for i in range(1, L1+1):
            for d in range(dim):
                if seq2[d, j-1] == 1:
                    p_max = F[i-1, j-1] + seq1[d, i-1]
                    break
            if (F[i-1, j] - gap) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - gap) > p_max:
                p_max = F[i, j-1] - gap
            F[i, j] = p_max
    return F


def F_trace_fast(double[:, :] F, double[:, :] seq1, double[:, :] seq2, int j_end, int gap):

    cdef int L_1 = seq1.shape[1]
    cdef int L_2 = seq2.shape[1]
    cdef int dim = 4
    cdef int i = L_1, j = j_end
    cdef int d, j_max, g_total = 0
    cdef double p_max = 0
    cdef double[:, :] matched_sub = np.zeros([4, L_1])

    while i != 0 and j != 0:
        for d in range(dim):
            if seq2[d, j-1] == 1:
                diag = F[i-1, j-1] + seq1[d, i-1]
                break
        if diag == F[i, j]:
            matched_sub[:, i - 1] = seq2[:, j - 1]
            i -= 1
            j -= 1
        elif F[i, j] == F[i, j - 1] - gap:
            j -= 1
            g_total += 1
        elif F[i, j] == F[i - 1, j] - gap:
            i -= 1
            g_total += 1
        else:
            matched_sub[:, i - 1] = seq2[:, j - 1]
            i -= 1
            j -= 1

    return matched_sub, g_total


def ASM_convert_fast(double[:, :] F, double[:, :] seq1, double[:, :] seq2, int gap):
    cdef int L_1 = seq1.shape[1]
    cdef int L_2 = seq2.shape[1]
    cdef int out_dim = L_2 + 1 - L_1
    cdef double[:, :, :] matched_array = np.zeros([out_dim, seq1.shape[0], seq1.shape[1]])
    cdef double[:, :] matched_sub
    cdef int g_total

    gap_array = np.zeros(out_dim)
    for j in range(L_1, L_2 + 1):
        matched_sub, g_total = F_trace_fast(F, seq1, seq2, j, gap)
        matched_array[j - L_1, :, :] = matched_sub
        gap_array[j - L_1] = g_total

    return matched_array, gap_array


def anchor_align_fast(double[:, :] seq1, double[:, :] seq2, int gap):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L1 + 1, L2 + 1])
    cdef int i, j, d
    cdef double p_max = 0, last_row_max = 0
    cdef int j_end
    nucleotides = 'ATCG'

    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        for i in range(1, L1+1):
            for d in range(dim):
                if seq2[d, j-1] == 1:
                    p_max = F[i-1, j-1] + seq1[d, i-1]
                    break
            if (F[i-1, j] - gap) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - gap) > p_max:
                p_max = F[i, j-1] - gap
            F[i, j] = p_max

        if F[L1, j] > last_row_max:
            last_row_max = F[L1, j]
            j_end = j

    i = L1
    j = j_end
    p_align = []
    s_align = []
    cdef double diag
    while i != 0 and j != 0:
        for d in range(dim):
            if seq2[d, j-1] == 1:
                diag = F[i-1, j-1] + seq1[d, i-1]
                break
        if diag == F[i, j]:
            i -= 1
            j -= 1
            p_align.append('*')
            s_align.append(nucleotides[d])
        elif F[i, j] == F[i, j - 1] - gap:
            j -= 1
            p_align.append('-')
            s_align.append(nucleotides[d])
        elif F[i, j] == F[i - 1, j] - gap:
            i -= 1
            p_align.append('*')
            s_align.append('-')
        else:
            i -= 1
            j -= 1
            p_align.append('*')
            s_align.append(nucleotides[d])

    if i > 0:
        p_align.append('*')
        s_align.append('-')
        i -= 1

    j_min = j
    p_align.reverse()
    s_align.reverse()

    return ''.join(p_align), ''.join(s_align), (j_min, j_end), last_row_max


def nw_align(s1, s2, gap):

    L1 = len(s1)
    L2 = len(s2)
    cdef long[:, :] F = np.zeros([L1 + 1, L2 + 1]).astype(int)
    cdef int i, j, p_max
    for i in range(L1+1):
        F[i, 0] = -1 * i
    for j in range(1, L2+1):
        F[0, j] = -1 * j
        for i in range(1, L1+1):
            p_max = F[i-1, j-1] + 2 * (s1[i-1] == s2[j - 1]) - 1
            if (F[i-1, j] - 1) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - 1) > p_max:
                p_max = F[i, j-1] - gap

            F[i, j] = p_max

    i = L1
    j = L2
    s_align = []
    ref_align = []
    cdef double diag
    while i + j != 0:

        if  i*j > 0:
            if F[i, j] == F[i - 1, j - 1] + 2 * (s1[i - 1] == s2[j - 1]) - 1:
                s_align.append(s1[i - 1])
                ref_align.append('*')
                i -= 1
                j -= 1
            elif F[i, j] == F[i - 1, j] - 1:
                s_align.append(s1[i - 1])
                ref_align.append('-')
                i -= 1
            else:
                s_align.append('-')
                ref_align.append('*')
                j -= 1

        elif i>0:
            s_align.append(s1[i - 1])
            ref_align.append('-')
            i -= 1
        elif j>0:
            s_align.append('-')
            ref_align.append('*')
            j -= 1


    s_align.reverse()
    ref_align.reverse()

    return ''.join(s_align), ''.join(ref_align)


def asm_align(s1, s_ref, int gap):

    L1 = len(s1)
    L2 = len(s_ref)
    cdef long[:, :] F = np.zeros([L1 + 1, L2 + 1]).astype(int)
    cdef int i, j, p_max
    cdef int last_row_max = -100

    for i in range(L1+1):
        F[i, 0] = -1 * i
    for j in range(1, L2+1):
        for i in range(1, L1+1):
            p_max = F[i-1, j-1] + 2 * (s1[i-1] == s_ref[j - 1]) - 1
            if (F[i-1, j] - 1) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - 1) > p_max:
                p_max = F[i, j-1] - gap

            F[i, j] = p_max

        if F[L1, j] > last_row_max:
            last_row_max = F[L1, j]

    i = L1
    j = L2
    s_align = []
    s_ref_align = []
    cdef double diag

    while F[L1, j] != last_row_max:
        s_align.append('-')
        s_ref_align.append('*')
        j -= 1

    while i + j != 0:

        if  i*j > 0:
            if F[i, j] == F[i - 1, j - 1] + 2 * (s1[i - 1] == s_ref[j - 1]) - 1:
                s_align.append(s1[i - 1])
                s_ref_align.append('*')
                i -= 1
                j -= 1
            elif F[i, j] == F[i - 1, j] - 1:
                s_align.append(s1[i - 1])
                s_ref_align.append('-')
                i -= 1
            elif F[i, j] == F[i, j - 1] - 1:
                s_align.append('-')
                s_ref_align.append('*')
                j -= 1
            else:
                s_align.append(s1[i - 1])
                s_ref_align.append('*')
                i -= 1
                j -= 1

        elif i > 0:
            s_align.append(s1[i - 1])
            s_ref_align.append('-')
            i -= 1
        elif j > 0:
            s_align.append('-')
            s_ref_align.append('*')
            j -= 1

    s_align.reverse()
    s_ref_align.reverse()

    return ''.join(s_align), ''.join(s_ref_align)



def nw_profile_align(double[:, :] seq1, double[:, :] seq2, int gap):
    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef double[:, :] F = np.zeros([L1 + 1, L2 + 1])
    cdef int i, j
    cdef double match, p_max
    cdef int dim = 4

    for i in range(L1+1):
        F[i, 0] = -1 * i
    for j in range(1, L2+1):
        F[0, j] = -1 * j
        for i in range(1, L1+1):
            match = 0
            for d in range(dim):
                match += seq1[d, i-1] * seq2[d, j-1]
            p_max = F[i-1, j-1] + match
            if (F[i-1, j] - 1) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - 1) > p_max:
                p_max = F[i, j-1] - gap

            F[i, j] = p_max

    i = L1
    j = L2
    s1_align = []
    s2_align = []
    while i + j != 0:

        if  i*j > 0:
            match = 0
            for d in range(dim):
                match += seq1[d, i - 1] * seq2[d, j - 1]

            if F[i, j] == F[i - 1, j - 1] + match:
                s1_align.append('*')
                s2_align.append('*')
                i -= 1
                j -= 1
            elif F[i, j] == F[i - 1, j] - 1:
                s1_align.append('*')
                s2_align.append('-')
                i -= 1
            elif F[i, j] == F[i, j - 1] - 1:
                s1_align.append('-')
                s2_align.append('*')
                j -= 1
            else:
                s1_align.append('*')
                s2_align.append('*')
                i -= 1
                j -= 1


        elif i>0:
            s1_align.append('*')
            s2_align.append('-')
            i -= 1
        elif j>0:
            s1_align.append('-')
            s2_align.append('*')
            j -= 1


    s1_align.reverse()
    s2_align.reverse()

    return ''.join(s1_align), ''.join(s2_align)


def asm_profile_align(double[:, :] seq1, double[:, :] seq2, int gap):

    cdef int L1 = seq1.shape[1]
    cdef int L2 = seq2.shape[1]
    cdef int dim = 4
    cdef double[:, :] F = np.zeros([L1 + 1, L2 + 1])
    cdef int i, j, d
    cdef double p_max = 0, last_row_max = 0, match
    cdef int j_end

    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        for i in range(1, L1+1):
            match = 0
            for d in range(dim):
                match += seq1[d, i - 1] * seq2[d, j - 1]
            p_max = F[i-1, j-1] + match
            if (F[i-1, j] - gap) > p_max:
                p_max = F[i-1, j] - gap
            if (F[i, j-1] - gap) > p_max:
                p_max = F[i, j-1] - gap
            F[i, j] = p_max

        if F[L1, j] > last_row_max:
            last_row_max = F[L1, j]
            j_end = j

    i = L1
    j = L2
    s_query_align = []
    s_ref_align = []

    while F[L1, j] - last_row_max > 1e-8:
        s_query_align.append('-')
        s_ref_align.append('*')
        j -= 1

    cdef double residue_diag, residue_up, residue_left
    # printf("%i, %i, %i \n", i, j, i + j)
    while i + j != 0:
        if i * j > 0:
            match = 0
            for d in range(dim):
                match += seq1[d, i - 1] * seq2[d, j - 1]

            residue_diag = abs(F[i, j] - (F[i - 1, j - 1] + match))
            residue_up = abs(F[i, j] - (F[i - 1, j] - gap))
            residue_left = abs(F[i, j] - (F[i, j - 1] - gap))

            if residue_up < 1e-8:
                s_query_align.append('*')
                s_ref_align.append('-')
                i -= 1
            elif residue_left < 1e-8:
                s_query_align.append('-')
                s_ref_align.append('*')
                j -= 1
            elif residue_diag < 1e-8:
                s_query_align.append('*')
                s_ref_align.append('*')
                i -= 1
                j -= 1

        elif i > 0:
            s_query_align.append('*')
            s_ref_align.append('-')
            i -= 1
        else:
            s_query_align.append('-')
            s_ref_align.append('*')
            j -= 1

    j_min = j
    s_query_align.reverse()
    s_ref_align.reverse()

    return ''.join(s_query_align), ''.join(s_ref_align)
