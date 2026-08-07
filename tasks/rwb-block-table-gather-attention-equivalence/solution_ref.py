import math


def paged_gqa_attention(k_phys, v_phys, block_table, q, n_kv_heads):
    """Gather logical KV from a paged block table and run GQA attention.

    k_phys, v_phys: (num_phys_blocks, block_size, n_kv_heads, D)
    block_table:    (L_b,) physical block index per logical position
    q:              (n_q_heads, D)
    """
    num_phys = len(k_phys)
    block_size = len(k_phys[0])
    H_kv = len(k_phys[0][0])
    D = len(k_phys[0][0][0])
    L_b = len(block_table)
    seq_len = L_b * block_size

    n_q_heads = len(q)
    group = n_q_heads // n_kv_heads
    scale = 1.0 / math.sqrt(D)

    out = [[0.0] * D for _ in range(n_q_heads)]
    for h in range(n_q_heads):
        kv_h = h // group
        scores = [0.0] * seq_len
        for i in range(seq_len):
            l = i // block_size
            b = i % block_size
            phys = int(block_table[l])
            dot = 0.0
            for d in range(D):
                dot += k_phys[phys][b][kv_h][d] * q[h][d]
            scores[i] = dot * scale

        max_score = scores[0]
        for i in range(1, seq_len):
            if scores[i] > max_score:
                max_score = scores[i]

        for i in range(seq_len):
            scores[i] -= max_score

        w = [0.0] * seq_len
        for i in range(seq_len):
            w[i] = math.exp(scores[i])

        sum_w = 0.0
        for i in range(seq_len):
            sum_w += w[i]

        for i in range(seq_len):
            w[i] /= sum_w

        for d in range(D):
            val_out = 0.0
            for i in range(seq_len):
                l = i // block_size
                b = i % block_size
                phys = int(block_table[l])
                val_out += w[i] * v_phys[phys][b][kv_h][d]
            out[h][d] = val_out

    return out
