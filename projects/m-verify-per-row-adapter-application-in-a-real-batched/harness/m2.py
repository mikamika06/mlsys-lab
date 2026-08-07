import ref


def check(workdir):
    from loraserving.memory import (
        compute_adapter_memory_bytes,
        compute_replica_memory_bytes,
        find_adapter_vs_replica_crossover,
    )

    out = {"memory_matches": 0.0, "crossover_matches": 0.0}

    test_configs = [
        {"num_adapters": 10, "rank": 16, "num_target_layers": 32, "hidden_size": 4096, "dtype_bytes": 2},
        {"num_adapters": 64, "rank": 8, "num_target_layers": 40, "hidden_size": 8192, "dtype_bytes": 2},
    ]

    mem_ok = True
    for cfg in test_configs:
        want_adapter = ref.ref_compute_adapter_memory_bytes(**cfg)
        got_adapter = compute_adapter_memory_bytes(**cfg)
        if want_adapter != got_adapter:
            mem_ok = False
            out["_note"] = f"adapter mem mismatch: want {want_adapter}, got {got_adapter}"
            break

    replica_cfg = {"num_replicas": 4, "base_params": 7_000_000_000, "kv_cache_bytes_per_replica": 2_000_000_000, "dtype_bytes": 2}
    want_replica = ref.ref_compute_replica_memory_bytes(**replica_cfg)
    got_replica = compute_replica_memory_bytes(**replica_cfg)
    if want_replica != got_replica:
        mem_ok = False
        out["_note"] = f"replica mem mismatch: want {want_replica}, got {got_replica}"

    if mem_ok:
        out["memory_matches"] = 1.0

    cross_cfg = {
        "rank": 16,
        "num_target_layers": 32,
        "hidden_size": 4096,
        "base_params": 1_000_000,
        "kv_cache_bytes": 500_000,
        "max_adapters": 100,
        "dtype_bytes": 2,
    }
    want_cross = ref.ref_find_adapter_vs_replica_crossover(**cross_cfg)
    got_cross = find_adapter_vs_replica_crossover(**cross_cfg)
    if want_cross == got_cross:
        out["crossover_matches"] = 1.0
    else:
        out["_note"] = f"crossover mismatch: want {want_cross}, got {got_cross}"

    return out
