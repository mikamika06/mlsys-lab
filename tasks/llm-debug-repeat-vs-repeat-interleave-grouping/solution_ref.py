def expand_kv_heads(kv: list[list[list[list[float]]]], num_q_heads: int) -> list[list[list[list[float]]]]:
    h_kv = len(kv[0])
    r = num_q_heads // h_kv
    out = []
    for batch in kv:
        new_batch = []
        for head in batch:
            for _ in range(r):
                new_batch.append([row[:] for row in head])
        out.append(new_batch)
    return out
