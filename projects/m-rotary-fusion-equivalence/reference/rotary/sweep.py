def optimal_num_splits(seq_len):
    if seq_len <= 512:
        return 1
    elif seq_len <= 2048:
        return 4
    else:
        return 8
