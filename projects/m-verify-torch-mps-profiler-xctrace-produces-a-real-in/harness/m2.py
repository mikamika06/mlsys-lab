import ref


def check(workdir):
    from msprofiler.metrics import compute_misattribution
    from msprofiler.compare import compare_framework_timings

    out = {"misattribution_match": 0.0, "ratio_match": 0.0}
    try:
        m_val = compute_misattribution(ref.CPU_DURATIONS, ref.MPS_DURATIONS)
        expected_m = sum(ref.CPU_DURATIONS) / (sum(ref.CPU_DURATIONS) + sum(ref.MPS_DURATIONS))
        if abs(m_val - expected_m) < 1e-5:
            out["misattribution_match"] = 1.0
        else:
            out["_note"] = f"Misattribution mismatch: got {m_val}, expected {expected_m}"

        comp = compare_framework_timings(ref.MLX_TIMES, ref.TORCH_TIMES)
        expected_speedup = (sum(ref.TORCH_TIMES) / len(ref.TORCH_TIMES)) / (sum(ref.MLX_TIMES) / len(ref.MLX_TIMES))
        if isinstance(comp, dict) and abs(comp.get("speedup", 0) - expected_speedup) < 1e-5:
            out["ratio_match"] = 1.0
        else:
            out["_note"] = f"Comparison mismatch: got {comp}"
    except Exception as e:
        out["_note"] = f"Error in metrics/compare: {type(e).__name__}: {str(e)[:100]}"
    return out
