def classify_causal_tiles(seq_len, block_size):
    num_blocks = seq_len // block_size
    result = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if j < i:
                row.append("full")
            elif j == i:
                row.append("diagonal")
            else:
                row.append("empty")
        result.append(row)
    return result
