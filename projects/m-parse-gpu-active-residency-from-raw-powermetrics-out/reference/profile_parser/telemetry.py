import numpy as np


def correlate_residency_with_throughput(
    gpu_residencies: list[float], tokens_per_sec: list[float]
) -> dict:
    """Correlate parsed GPU residency with measured tokens/sec across model sizes."""
    if len(gpu_residencies) != len(tokens_per_sec) or len(gpu_residencies) == 0:
        return {"mean_residency": 0.0, "mean_tps": 0.0, "correlation": 0.0, "efficiency": 0.0}

    arr_res = np.array(gpu_residencies, dtype=np.float64)
    arr_tps = np.array(tokens_per_sec, dtype=np.float64)

    mean_res = float(np.mean(arr_res))
    mean_tps = float(np.mean(arr_tps))

    std_res = np.std(arr_res)
    std_tps = np.std(arr_tps)

    if std_res < 1e-9 or std_tps < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(arr_res, arr_tps)[0, 1])

    efficiency = mean_res / mean_tps if mean_tps > 0 else 0.0

    return {
        "mean_residency": mean_res,
        "mean_tps": mean_tps,
        "correlation": corr,
        "efficiency": efficiency,
    }


def estimate_ane_utilization(
    ane_powers_mw: list[float], max_ane_power_mw: float = 8000.0
) -> dict:
    """Extract ANE power stats and estimate utilization relative to peak capacity."""
    if not ane_powers_mw:
        return {"avg_power_mw": 0.0, "peak_power_mw": 0.0, "estimated_utilization_pct": 0.0}

    arr_p = np.array(ane_powers_mw, dtype=np.float64)
    avg_p = float(np.mean(arr_p))
    peak_p = float(np.max(arr_p))

    util_pct = (avg_p / max_ane_power_mw) * 100.0 if max_ane_power_mw > 0 else 0.0

    return {
        "avg_power_mw": avg_p,
        "peak_power_mw": peak_p,
        "estimated_utilization_pct": min(100.0, max(0.0, util_pct)),
    }
