import numpy as np


def compute_imbalance_over_time(log_entries):
    timestamps = []
    imbalance_ratios = []
    eplb_effective_ratios = []

    for entry in log_entries:
        ts = float(entry["timestamp"])
        raw_tokens = np.array(entry["raw_expert_tokens"], dtype=np.float64)
        eplb_tokens = np.array(entry["eplb_expert_tokens"], dtype=np.float64)

        raw_max = np.max(raw_tokens)
        raw_mean = np.mean(raw_tokens)
        raw_ratio = float(raw_max / raw_mean) if raw_mean > 0 else 1.0

        eplb_max = np.max(eplb_tokens)
        eplb_mean = np.mean(eplb_tokens)
        eplb_ratio = float(eplb_max / eplb_mean) if eplb_mean > 0 else 1.0

        timestamps.append(ts)
        imbalance_ratios.append(raw_ratio)
        eplb_effective_ratios.append(eplb_ratio)

    return {
        "timestamps": timestamps,
        "imbalance_ratios": imbalance_ratios,
        "eplb_effective_ratios": eplb_effective_ratios,
        "mean_imbalance": float(np.mean(imbalance_ratios)) if imbalance_ratios else 0.0,
    }
