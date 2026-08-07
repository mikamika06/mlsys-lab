def compute_kernel_intensity(flops: float, bytes_transferred: float) -> float:
    if bytes_transferred <= 0:
        return 0.0
    return float(flops) / float(bytes_transferred)


def aggregate_profile(records: list[dict]) -> dict[str, dict]:
    aggregated = {}
    for rec in records:
        name = rec["name"]
        flops = float(rec["flops"])
        b = float(rec["bytes"])
        t = float(rec["time_us"])
        if name not in aggregated:
            aggregated[name] = {
                "total_flops": 0.0,
                "total_bytes": 0.0,
                "total_time_us": 0.0,
                "count": 0
            }
        aggregated[name]["total_flops"] += flops
        aggregated[name]["total_bytes"] += b
        aggregated[name]["total_time_us"] += t
        aggregated[name]["count"] += 1

    for name, stats in aggregated.items():
        tf = stats["total_flops"]
        tb = stats["total_bytes"]
        tt = stats["total_time_us"]
        stats["intensity"] = tf / tb if tb > 0 else 0.0
        stats["achieved_tflops"] = (tf / (tt * 1e-6) / 1e12) if tt > 0 else 0.0

    return aggregated


def model_total_stats(aggregated: dict[str, dict]) -> dict:
    total_flops = sum(s["total_flops"] for s in aggregated.values())
    total_bytes = sum(s["total_bytes"] for s in aggregated.values())
    total_time_us = sum(s["total_time_us"] for s in aggregated.values())
    intensity = total_flops / total_bytes if total_bytes > 0 else 0.0
    achieved_tflops = (total_flops / (total_time_us * 1e-6) / 1e12) if total_time_us > 0 else 0.0
    return {
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "total_time_us": total_time_us,
        "overall_intensity": intensity,
        "overall_achieved_tflops": achieved_tflops
    }
