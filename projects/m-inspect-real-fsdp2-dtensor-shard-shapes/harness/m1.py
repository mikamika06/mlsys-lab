import ref


def check(workdir):
    from fsdpshards.sharding import get_dtensor_shard_info

    out = {"shards_matched": 0.0, "total_cases": float(len(ref.M1_CASES))}
    matched = 0
    for i, (shape, mesh_size, rank, shard_dim) in enumerate(ref.M1_CASES):
        want = ref.get_dtensor_shard_info(shape, mesh_size, rank, shard_dim)
        got = get_dtensor_shard_info(shape, mesh_size, rank, shard_dim)
        if got == want:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["shards_matched"] = float(matched)
    return out
