def expand_kv_heads(kv: list[list[list[list[float]]]], n_query_heads: int) -> list[list[list[list[float]]]]:
    batch_size = len(kv)
    kv_heads = len(kv[0])
    seq_len = len(kv[0][0])
    dim = len(kv[0][0][0])

    repeat = n_query_heads // kv_heads

    out = []
    for b in range(batch_size):
        b_list = []
        for h in range(kv_heads):
            for _ in range(repeat):
                head_data = []
                for s in range(seq_len):
                    seq_data = []
                    for d in range(dim):
                        seq_data.append(float(kv[b][h][s][d]))
                    head_data.append(seq_data)
                b_list.append(head_data)
        out.append(b_list)
    return out
