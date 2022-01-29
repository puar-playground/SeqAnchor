# from msa_utility import msa_compress_gap


def read_seq(seq_dir):
    f_s = open(seq_dir, 'r')
    seq_list = []
    while 1:
        name = f_s.readline()[1:-1]
        if not name:
            break
        seq = f_s.readline()[:-1]
        seq_list.append(seq)
    return seq_list


def match_lookup(c1, c2):
    if c1 == c2:
        if c1 != '-':
            return 1
        else:
            return 0
    else:
        return -1


def aop(msa_list):
    n_seq = len(msa_list)
    score_all = 0
    n_column = len(msa_list[0])

    for c in range(n_column):
        score_column = 0
        print('\rcolumn: %i, score: %4.2f' % (c, score_all), end='')
        cnt = 0
        for i in range(n_seq - 1):
            for j in range(i + 1, n_seq):
                # print(msa_list[i][c], msa_list[j][c])
                score_column += match_lookup(msa_list[i][c], msa_list[j][c])
                cnt += 1
        score_all += score_column / cnt

    print('\rfinished. score: %4.2f' % score_all)
    return score_all


def sop(msa_list):
    n_seq = len(msa_list)
    score_all = 0
    n_column = len(msa_list[0])

    for c in range(n_column):
        score_column = 0
        print('\rcolumn: %i, score: %4.2f' % (c, score_all), end='')
        for i in range(n_seq - 1):
            for j in range(i + 1, n_seq):
                # print(msa_list[i][c], msa_list[j][c])
                score_column += match_lookup(msa_list[i][c], msa_list[j][c])
        score_all += score_column

    print('\rfinished. score: %4.2f' % score_all)
    return score_all


if __name__ == "__main__":

    msa = read_seq('/Users/chenjian/Desktop/muscle/gg/gg_' +
                   str(4) + '_aligned.fa')
    msa = msa[:100]

    # msa = msa_compress_gap(msa)
    socre = aop(msa)

    for i in range(10):
        print(msa[i])
        # print(len(msa[i]))
