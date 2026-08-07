def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from planner.calculator import MemoryPlanner

    m = {"advise_ok": 0.0}
    cfg = {"num_params": 7000000000, "micro_batch_size": 8, "zero_stage": 0, "activation_checkpointing": False, "cpu_offload": False}
    p = MemoryPlanner(cfg)

    limit = p.total_memory() // 2
    advice = p.advise(limit)

    if isinstance(advice, list) and len(advice) > 0 and "activation_checkpointing" in advice:
        m["advise_ok"] = 1.0

    return m
