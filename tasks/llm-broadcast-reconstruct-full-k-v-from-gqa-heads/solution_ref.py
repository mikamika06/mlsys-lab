def expand_gqa_kv(kv: list[list[list[list[float]]]], num_query_heads: int) -> list[list[list[list[float]]]]:
    B = len(kv)
    n_kv = len(kv[0])
    repeat = num_query_heads // n_kv

    result = []
    for b in range(B):
        batch_heads = []
        for i in range(n_kv):
            head_data = kv[b][i]
            for _ in range(repeat):
                batch_heads.append([list(s) for s in head_data])
        result.append(batch_heads)
    return result
