import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"latency_ratio_valid": 0.0, "cache_hit_ratio": 0.0}

    try:
        from triton_cache.cache_demo import measure_compile_vs_hit_latency
        from triton_cache.kernel import mock_triton_kernel

        cold, warm = measure_compile_vs_hit_latency(mock_triton_kernel, 128)

        if cold > 0 and warm > 0 and cold > warm:
            out["latency_ratio_valid"] = 1.0

        ratio = cold / warm if warm > 0 else 0.0
        out["cache_hit_ratio"] = float(ratio)

    except Exception as e:
        out["_note"] = f"Milestone 2 check failed with error: {type(e).__name__}: {e}"

    return out
