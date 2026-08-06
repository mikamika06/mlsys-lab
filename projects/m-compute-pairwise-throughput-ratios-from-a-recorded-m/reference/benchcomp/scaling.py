import numpy as np


def compute_scaling_efficiency(records):
    """Compute mean scaling efficiency per framework and GPU count relative to single-GPU baseline."""
    baselines = {}
    for r in records:
        if r["num_gpus"] == 1:
            key = (r["framework"], r["config_id"])
            baselines[key] = r["tokens_per_sec"]

    effs = {}
    for r in records:
        key_base = (r["framework"], r["config_id"])
        if key_base in baselines and r["num_gpus"] > 1:
            base_tps = baselines[key_base]
            pair_key = (r["framework"], r["num_gpus"])
            eff = r["tokens_per_sec"] / (base_tps * r["num_gpus"])
            effs.setdefault(pair_key, []).append(eff)

    return {pair_key: float(np.mean(vals)) for pair_key, vals in effs.items()}
