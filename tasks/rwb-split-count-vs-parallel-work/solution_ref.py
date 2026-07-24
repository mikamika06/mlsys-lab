def choose_split_count(batch, query_rows, kv_len, split_counts):
    works = []
    for s in split_counts:
        parallel_tiles = batch * query_rows * s
        combine_cost = s * kv_len
        works.append(float(parallel_tiles - combine_cost))

    best = split_counts[0]
    best_value = works[0]
    for s, value in zip(split_counts[1:], works[1:]):
        if value > best_value:
            best = s
            best_value = value

    return works, best
