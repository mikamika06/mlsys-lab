def grade(sol, fx) -> dict:
    """
    Grader for rwb-kv-memory-for-n-contiguous-slots.
    Computes the reference value using the same arithmetic as the spec,
    then compares it to the candidate's output.
    """
    test_cases = [
        (
            {"layers": 12, "kv_heads": 12, "head_dim": 64, "dtype_bytes": 4, "n_ctx": 2048},
            3
        ),
        (
            {"layers": 24, "kv_heads": 16, "head_dim": 128, "dtype_bytes": 2, "n_ctx": 1024},
            5
        ),
        (
            {"layers": 6, "kv_heads": 8, "head_dim": 32, "dtype_bytes": 1, "n_ctx": 512},
            10
        ),
    ]
    ok = 1.0
    for cfg, n_slots in test_cases:
        try:
            got = sol.kv_memory_bytes(cfg, n_slots)
        except Exception:
            return {"exact_match": 0.0}
        ref = (
            n_slots
            * 2
            * cfg["layers"]
            * cfg["kv_heads"]
            * cfg["head_dim"]
            * cfg["n_ctx"]
            * cfg["dtype_bytes"]
        )
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
