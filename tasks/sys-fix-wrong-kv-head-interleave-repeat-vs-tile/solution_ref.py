def expand_kv_heads(kv: list[list[list[list[float]]]], num_query_heads: int) -> list[list[list[list[float]]]]:
    batch_size = len(kv)
    num_kv_heads = len(kv[0])
    seq_len = len(kv[0][0])
    dim = len(kv[0][0][0])

    repeat = num_query_heads // num_kv_heads

    out = []
    for b in range(batch_size):
        b_list = []
        for h_kv in range(num_kv_heads):
            for _ in range(repeat):
                h_list = []
                for t in range(seq_len):
                    d_list = []
                    for d in range(dim):
                        d_list.append(kv[b][h_kv][t][d])
                    h_list.append(d_list)
                b_list.append(h_list)
        out.append(b_list)
    return out
