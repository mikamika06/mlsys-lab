import numpy as np


def measure_latency(raw_runs, warmup_count=1):
    if len(raw_runs) <= warmup_count:
        valid = raw_runs
    else:
        valid = raw_runs[warmup_count:]
    arr = np.array(valid, dtype=float)
    med = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return {"median": med, "iqr": iqr}


def split_phases(prefill_times, decode_times):
    p_arr = np.array(prefill_times, dtype=float)
    d_arr = np.array(decode_times, dtype=float)
    return {
        "prefill": {"mean": float(np.mean(p_arr)), "std": float(np.std(p_arr))},
        "decode": {"mean": float(np.mean(d_arr)), "std": float(np.std(d_arr))}
    }


def required_samples(std_dev, target_width, confidence_z=1.96):
    n = (2 * confidence_z * std_dev / target_width) ** 2
    return int(np.ceil(n))


def correct_thermal(raw_speeds, degradation_rate=0.01):
    corrected = []
    for i, speed in enumerate(raw_speeds):
        factor = 1.0 + i * degradation_rate
        corrected.append(float(speed * factor))
    return corrected


def check_consistency(run_intervals):
    if not run_intervals:
        return True
    lows = [iv[0] for iv in run_intervals]
    highs = [iv[1] for iv in run_intervals]
    max_low = max(lows)
    min_high = min(highs)
    return max_low <= min_high


def generate_ci_report(metrics_dict):
    return {"status": "success", "metrics": metrics_dict}
