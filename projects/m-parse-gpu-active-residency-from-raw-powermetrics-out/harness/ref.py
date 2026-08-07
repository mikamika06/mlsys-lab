import numpy as np


def generate_powermetrics_fixture():
    samples_text = []
    expected_gpu = []
    expected_ane = []

    configs = [
        ("GPU HW active residency: 78.40%", 78.40, "ANE Power: 120 mW", 120.0),
        ("GPU HW idle residency: 12.50%", 87.50, "ANE Power: 2.5 W", 2500.0),
        ("GPU active frequency: 1200 MHz (65.20%)", 65.20, "ANE Power: 0 mW", 0.0),
        ("GPU use: 42.10%", 42.10, "ANE Power: 450 mW", 450.0),
        ("GPU HW active residency: 95.00%", 95.00, "ANE Power: 1.2 W", 1200.0),
    ]

    for gpu_str, gpu_val, ane_str, ane_val in configs:
        sample = f"*** Sampled system activity\n{gpu_str}\n{ane_str}\nCPU Power: 1500 mW\n"
        samples_text.append(sample)
        expected_gpu.append(gpu_val)
        expected_ane.append(ane_val)

    text = "\n".join(samples_text)
    return text, expected_gpu, expected_ane


def reference_correlate(gpu_residencies, tokens_per_sec):
    arr_res = np.array(gpu_residencies, dtype=np.float64)
    arr_tps = np.array(tokens_per_sec, dtype=np.float64)
    mean_res = float(np.mean(arr_res))
    mean_tps = float(np.mean(arr_tps))
    std_res = np.std(arr_res)
    std_tps = np.std(arr_tps)
    corr = float(np.corrcoef(arr_res, arr_tps)[0, 1]) if std_res > 1e-9 and std_tps > 1e-9 else 0.0
    eff = mean_res / mean_tps if mean_tps > 0 else 0.0
    return {
        "mean_residency": mean_res,
        "mean_tps": mean_tps,
        "correlation": corr,
        "efficiency": eff,
    }


def reference_ane_utilization(ane_powers_mw, max_power_mw=8000.0):
    arr_p = np.array(ane_powers_mw, dtype=np.float64)
    avg_p = float(np.mean(arr_p))
    peak_p = float(np.max(arr_p))
    util = (avg_p / max_power_mw) * 100.0 if max_power_mw > 0 else 0.0
    return {
        "avg_power_mw": avg_p,
        "peak_power_mw": peak_p,
        "estimated_utilization_pct": min(100.0, max(0.0, util)),
    }


MODEL_RUNS = {
    "model_3b": {"gpu_residency": [45.0, 48.0, 42.0, 46.0, 44.0], "tps": [120.0, 125.0, 118.0, 122.0, 121.0]},
    "model_7b": {"gpu_residency": [78.0, 82.0, 80.0, 85.0, 79.0], "tps": [55.0, 58.0, 56.0, 60.0, 54.0]},
    "model_13b": {"gpu_residency": [95.0, 98.0, 96.0, 97.0, 99.0], "tps": [28.0, 29.0, 27.5, 28.5, 29.5]},
}
