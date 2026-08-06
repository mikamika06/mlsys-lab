import ref


def check(workdir):
    from ggufsplit.planner import compute_split_plan
    from ggufsplit.sizing import predict_shard_sizes

    out = {"sizes_match": 0.0, "exact_match": 0.0}
    tensors, max_size = ref.CASES[0]
    shards = compute_split_plan(tensors, max_size)
    got_sizes = predict_shard_sizes(shards)
    expected_sizes = [sum(sz for _, sz in s) for s in shards]

    if got_sizes == expected_sizes:
        out["sizes_match"] = 1.0
        out["exact_match"] = 1.0
    else:
        out["_note"] = f"got sizes {got_sizes}, expected {expected_sizes}"
    return out
