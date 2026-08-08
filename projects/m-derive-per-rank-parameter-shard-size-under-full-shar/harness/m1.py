import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from fsdp_analyzer.sharding import (
        compute_rank_shard_size,
        compute_world_shard_distribution,
    )

    out = {"shards_matched": 0.0}
    ok = 0

    for cfg in ref.CONFIGS:
        num_params = cfg["num_params"]
        world_size = cfg["world_size"]

        want_dist = ref.ref_compute_world_shard_distribution(num_params, world_size)
        got_dist = compute_world_shard_distribution(num_params, world_size)

        ranks_ok = True
        for rank in range(world_size):
            want_rank = ref.ref_compute_rank_shard_size(num_params, world_size, rank)
            got_rank = compute_rank_shard_size(num_params, world_size, rank)
            if want_rank != got_rank:
                ranks_ok = False
                break

        if ranks_ok and want_dist == got_dist:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cfg {cfg}: got dist {got_dist[:3]}, expected {want_dist[:3]}"

    out["shards_matched"] = float(ok)
    return out
