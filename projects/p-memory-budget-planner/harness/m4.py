def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    from planner.calculator import MemoryPlanner

    m = {"accurate_total": 0.0}
    cfg = {"num_params": 7000000000, "hidden_size": 4096, "num_layers": 32, "seq_len": 4096, "micro_batch_size": 4, "bytes_per_param": 2, "world_size": 8, "zero_stage": 3, "activation_checkpointing": True, "cpu_offload": True}
    p = MemoryPlanner(cfg)

    expected = ref.oracle_total(cfg)
    actual = p.total_memory()

    if expected > 0 and abs(actual - expected) / expected <= 0.1:
        m["accurate_total"] = 1.0

    return m
