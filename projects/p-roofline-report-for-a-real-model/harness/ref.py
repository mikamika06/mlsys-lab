import random


def get_hw_spec():
    return {
        "peak_flops_per_sec": 312e12,
        "peak_bandwidth_bytes_sec": 2e12
    }


def generate_raw_profile(seed=42):
    rng = random.Random(seed)
    kernels = [
        ("qkv_proj", 1e9, 2e8, 120.0, 12),
        ("attention_sdpa", 5e9, 1e8, 80.0, 12),
        ("layernorm", 1e7, 5e7, 40.0, 24),
        ("mlp_gate_up", 4e9, 3e8, 150.0, 12),
        ("gelu_act", 2e7, 4e7, 35.0, 12),
        ("mlp_down", 4e9, 3e8, 150.0, 12)
    ]
    records = []
    for name, base_f, base_b, base_t, count in kernels:
        for _ in range(count):
            f_noise = rng.uniform(0.98, 1.02)
            b_noise = rng.uniform(0.98, 1.02)
            t_noise = rng.uniform(0.98, 1.02)
            records.append({
                "name": name,
                "flops": base_f * f_noise,
                "bytes": base_b * b_noise,
                "time_us": base_t * t_noise
            })
    return records


def generate_optimized_profile_pair(seed=46):
    init_raw = generate_raw_profile(seed)
    hw = get_hw_spec()
    target_kernels = ["layernorm", "gelu_act"]
    mem_red = 0.5
    eff = 1.0

    rng = random.Random(seed + 100)
    opt_raw = []
    for rec in init_raw:
        if rec["name"] in target_kernels:
            new_b = rec["bytes"] * (1.0 - mem_red)
            new_intensity = rec["flops"] / new_b if new_b > 0 else 0.0
            ceil = oracle_roofline_ceiling(new_intensity, hw)
            calc_t = (rec["flops"] / ceil) * 1e6
            min_bw_t = (new_b / hw["peak_bandwidth_bytes_sec"]) * 1e6
            pred_t = max(calc_t, min_bw_t)
            actual_t = pred_t * rng.uniform(0.98, 1.01)
            opt_raw.append({
                "name": rec["name"],
                "flops": rec["flops"],
                "bytes": new_b,
                "time_us": actual_t
            })
        else:
            opt_raw.append({
                "name": rec["name"],
                "flops": rec["flops"],
                "bytes": rec["bytes"],
                "time_us": rec["time_us"] * rng.uniform(0.99, 1.01)
            })

    opt_params = {
        "target_kernels": target_kernels,
        "memory_reduction_factor": mem_red,
        "target_efficiency": eff
    }
    return init_raw, opt_params, opt_raw


def oracle_compute_kernel_intensity(flops, bytes_transferred):
    if bytes_transferred <= 0:
        return 0.0
    return float(flops) / float(bytes_transferred)


def oracle_aggregate_profile(records):
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


def oracle_model_total_stats(aggregated):
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


def oracle_roofline_ceiling(intensity, hw_spec):
    peak_flops = float(hw_spec["peak_flops_per_sec"])
    peak_bw = float(hw_spec["peak_bandwidth_bytes_sec"])
    mem_bound_ceiling = peak_bw * float(intensity)
    return min(peak_flops, mem_bound_ceiling)


def oracle_classify_kernel(intensity, hw_spec):
    ridge_point = float(hw_spec["peak_flops_per_sec"]) / float(hw_spec["peak_bandwidth_bytes_sec"])
    if float(intensity) < ridge_point:
        return "memory_bound"
    return "compute_bound"


def oracle_kernel_performance_bound(kernel_stats, hw_spec):
    intensity = float(kernel_stats["intensity"])
    ceiling_flops_sec = oracle_roofline_ceiling(intensity, hw_spec)
    bound_type = oracle_classify_kernel(intensity, hw_spec)
    total_flops = float(kernel_stats["total_flops"])
    total_time_us = float(kernel_stats["total_time_us"])
    min_time_us = (total_flops / ceiling_flops_sec * 1e6) if ceiling_flops_sec > 0 else 0.0
    achieved_flops_sec = (total_flops / (total_time_us * 1e-6)) if total_time_us > 0 else 0.0
    efficiency = achieved_flops_sec / ceiling_flops_sec if ceiling_flops_sec > 0 else 0.0
    headroom_speedup = total_time_us / min_time_us if min_time_us > 0 else 1.0
    return {
        "ceiling_flops_sec": ceiling_flops_sec,
        "bound_type": bound_type,
        "min_time_us": min_time_us,
        "achieved_flops_sec": achieved_flops_sec,
        "efficiency": efficiency,
        "headroom_speedup": headroom_speedup
    }


def oracle_estimate_optimization_speedup(
    aggregated, hw_spec, target_kernels=None, memory_reduction_factor=0.0, target_efficiency=1.0
):
    original_total_time = sum(s["total_time_us"] for s in aggregated.values())
    predicted_total_time = 0.0
    for name, stats in aggregated.items():
        if target_kernels is None or name in target_kernels:
            flops = stats["total_flops"]
            orig_bytes = stats["total_bytes"]
            new_bytes = orig_bytes * (1.0 - memory_reduction_factor)
            new_intensity = flops / new_bytes if new_bytes > 0 else stats["intensity"]
            ceiling = oracle_roofline_ceiling(new_intensity, hw_spec)
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


def oracle_validate_prediction_against_profile(predicted_stats, actual_records):
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


def oracle_generate_prioritized_report(aggregated, hw_spec):
    total_model_time = sum(s["total_time_us"] for s in aggregated.values())
    report = []
    for name, stats in aggregated.items():
        bounds = oracle_kernel_performance_bound(stats, hw_spec)
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
