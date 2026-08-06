def compute_cost_efficiency(logs, gpu_hourly_cost):
    result = {}
    gpu_cost_per_sec = float(gpu_hourly_cost) / 3600.0

    by_mode = {}
    for entry in logs:
        m = entry["mode"]
        if m not in by_mode:
            by_mode[m] = {"tokens": 0, "time_sec": 0.0}
        by_mode[m]["tokens"] += int(entry["total_useful_tokens"])
        by_mode[m]["time_sec"] += float(entry["execution_time_sec"])

    for mode, data in by_mode.items():
        cost = data["time_sec"] * gpu_cost_per_sec
        tpd = (float(data["tokens"]) / cost) if cost > 0 else 0.0
        result[mode] = {
            "total_tokens": data["tokens"],
            "total_cost_dollars": float(cost),
            "tokens_per_dollar": float(tpd),
        }

    return result
