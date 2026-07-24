def _oracle(batch, query_rows, kv_len, split_counts):
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


def grade(sol, fx) -> dict:
    cases = [
        (1, 32, 64, [1, 2, 4, 8]),
        (4, 16, 128, [1, 3, 6]),
        (8, 64, 512, [2, 4, 8, 16]),
        (2, 7, 14, [1, 2, 3, 5]),
        (16, 1, 16, [1, 2, 4]),
    ]
    ok = 1.0
    for batch, query_rows, kv_len, split_counts in cases:
        expected = _oracle(batch, query_rows, kv_len, split_counts)
        try:
            got_values, got_best = sol.choose_split_count(
                batch, query_rows, kv_len, list(split_counts)
            )
            got = (list(got_values), got_best)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
