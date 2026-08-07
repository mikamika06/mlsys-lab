from roofline.model import roofline_ceiling, kernel_performance_bound


def estimate_optimization_speedup(
    aggregated: dict[str, dict],
    hw_spec: dict,
    target_kernels: list[str] = None,
    memory_reduction_factor: float = 0.0,
    target_efficiency: float = 1.0
) -> dict:
    original_total_time = sum(s["total_time_us"] for s in aggregated.values())
    predicted_total_time = 0.0
    for name, stats in aggregated.items():
        if target_kernels is None or name in target_kernels:
            flops = stats["total_flops"]
            orig_bytes = stats["total_bytes"]
            new_bytes = orig_bytes * (1.0 - memory_reduction_factor)
            new_intensity = flops / new_bytes if new_bytes > 0 else stats["intensity"]
            ceiling = roofline_ceiling(new_intensity, hw_spec)
            target_flops_sec = ceiling * target_efficiency
            if target_flops_sec > 0:
                calc_time_us = (flops / target_flops_sec) * 1e6
                peak_bw = hw_spec["peak_bandwidth_bytes_sec"]
                min_bw_time_us = (new_bytes / peak_bw) * 1e6
                kernel_pred_time = max(calc_time_us, min_bw_time_us)
            else:
                kernel_pred_time = stats["total_time_us"]
            predicted_total_time += kernel_pred_time
        else:
            predicted_total_time += stats["total_time_us"]
    time_saved_us = original_total_time - predicted_total_time
    speedup = original_total_time / predicted_total_time if predicted_total_time > 0 else 1.0
    return {
        "original_time_us": original_total_time,
        "predicted_time_us": predicted_total_time,
        "time_saved_us": time_saved_us,
        "speedup": speedup
    }


def validate_prediction_against_profile(predicted_stats: dict, actual_records: list[dict]) -> dict:
    actual_time_us = sum(r["time_us"] for r in actual_records)
    pred_time_us = float(predicted_stats["predicted_time_us"])
    relative_error = abs(pred_time_us - actual_time_us) / actual_time_us if actual_time_us > 0 else 0.0
    matches = relative_error <= 0.05
    return {
        "predicted_time_us": pred_time_us,
        "actual_time_us": actual_time_us,
        "relative_error": relative_error,
        "matches": matches
    }


def generate_prioritized_report(aggregated: dict[str, dict], hw_spec: dict) -> list[dict]:
    total_model_time = sum(s["total_time_us"] for s in aggregated.values())
    report = []
    for name, stats in aggregated.items():
        bounds = kernel_performance_bound(stats, hw_spec)
        current_time = stats["total_time_us"]
        min_time = bounds["min_time_us"]
        potential_savings = max(0.0, current_time - min_time)
        time_share_pct = (current_time / total_model_time * 100.0) if total_model_time > 0 else 0.0
        report.append({
            "name": name,
            "current_time_us": current_time,
            "min_time_us": min_time,
            "potential_savings_us": potential_savings,
            "headroom_speedup": bounds["headroom_speedup"],
            "bound_type": bounds["bound_type"],
            "time_share_pct": time_share_pct
        })
    report.sort(key=lambda item: item["potential_savings_us"], reverse=True)
    return report
