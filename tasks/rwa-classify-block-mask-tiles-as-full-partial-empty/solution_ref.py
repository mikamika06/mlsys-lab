import math


def classify_block_mask_tiles(mask_mod, seq_len_q, seq_len_kv, block_q, block_kv):
    num_q = seq_len_q // block_q
    num_kv = seq_len_kv // block_kv
    result = []
    for i in range(num_q):
        row = []
        for j in range(num_kv):
            vals = []
            for q in range(i * block_q, (i + 1) * block_q):
                for k in range(j * block_kv, (j + 1) * block_kv):
                    vals.append(bool(mask_mod(0, 0, q, k)))
            if all(vals):
                row.append("full")
            elif not any(vals):
                row.append("empty")
            else:
                row.append("partial")
        result.append(row)
    return result
