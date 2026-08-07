import ref


def check(workdir):
    from dcp.shard import compute_rank_file_size

    configs = ref.get_test_configs()
    matched = 0
    for cfg in configs:
        shapes = cfg["shapes"]
        for ws in cfg["world_sizes"]:
            oracle = ref.compute_oracle_file_sizes(shapes, ws)
            learner = [compute_rank_file_size(shapes, ws, r) for r in range(ws)]
            if oracle == learner:
                matched += 1
    return {"sizes_matched": float(matched)}
