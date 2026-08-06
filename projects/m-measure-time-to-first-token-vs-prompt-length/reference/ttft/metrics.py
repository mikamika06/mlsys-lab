import numpy as np


def extract_ttft(logs):
    ttfts = []
    for log in logs:
        if "prompt_eval_time_ms" in log and "tokens_evaluated" in log:
            ttfts.append((log["tokens_evaluated"], log["prompt_eval_time_ms"]))
    return ttfts


def aggregate_runs(runs):
    mapping = {}
    for length, time_ms in runs:
        mapping.setdefault(length, []).append(time_ms)
    result = []
    for length in sorted(mapping.keys()):
        times = mapping[length]
        result.append({
            "prompt_len": length,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times) if len(times) > 1 else 0.0),
        })
    return result
