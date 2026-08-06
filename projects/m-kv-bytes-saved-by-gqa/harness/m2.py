import ref


def check(workdir):
    try:
        from gqa_opt.memory import (
            analyze_gpu_expansion_overhead,
            calculate_kv_cache_bytes,
        )
    except Exception as e:
        return {
            "bytes_saved_rel_err": 1.0,
            "expansion_ratio_correct": 0.0,
            "_note": f"Import error: {e}",
        }

    test_params = [
        (ref.CONFIGS[1], 1, 2048, 2),
        (ref.CONFIGS[1], 8, 4096, 2),
        (ref.CONFIGS[2], 4, 8192, 2),
        (ref.CONFIGS[3], 16, 1024, 1),
    ]

    max_rel_err = 0.0
    exp_ratio_ok = True

    for cfg, bs, seq, dt in test_params:
        want_mem = ref.calculate_kv_cache_bytes(cfg, bs, seq, dt)
        want_exp = ref.analyze_gpu_expansion_overhead(cfg, bs, seq, dt)

        try:
            got_mem = calculate_kv_cache_bytes(cfg, bs, seq, dt)
            got_exp = analyze_gpu_expansion_overhead(cfg, bs, seq, dt)
        except Exception as e:
            return {
                "bytes_saved_rel_err": 1.0,
                "expansion_ratio_correct": 0.0,
                "_note": f"Exception during execution: {e}",
            }

        want_saved = want_mem["bytes_saved"]
        got_saved = got_mem.get("bytes_saved", -1)
        rel_err = abs(want_saved - got_saved) / max(1, want_saved)
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if (
            abs(
                got_exp.get("expansion_factor", 0)
                - want_exp["expansion_factor"]
            )
            > 1e-6
        ):
            exp_ratio_ok = False

    return {
        "bytes_saved_rel_err": float(max_rel_err),
        "expansion_ratio_correct": 1.0 if exp_ratio_ok else 0.0,
    }
