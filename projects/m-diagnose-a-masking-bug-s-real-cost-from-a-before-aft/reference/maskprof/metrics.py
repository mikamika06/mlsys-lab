def calculate_masking_cost(diff: dict, hw_spec: dict) -> dict:
    b_time = diff.get("before_time_ns", 0)
    a_time = diff.get("after_time_ns", 0)
    b_dram = diff.get("before_dram_bytes", 0)
    a_dram = diff.get("after_dram_bytes", 0)
    wasted_dram = diff.get("total_dram_wasted_bytes", 0)
    peak_bw = hw_spec.get("peak_dram_bw_gbs", 1.0)

    speedup = b_time / a_time if a_time > 0 else 1.0
    wasted_dram_bw_gbs = wasted_dram / b_time if b_time > 0 else 0.0
    wasted_bw_pct = (wasted_dram_bw_gbs / peak_bw) * 100.0 if peak_bw > 0 else 0.0

    eff_bw_before = b_dram / b_time if b_time > 0 else 0.0
    eff_bw_after = a_dram / a_time if a_time > 0 else 0.0

    time_saved_us = (b_time - a_time) / 1000.0

    return {
        "speedup": speedup,
        "wasted_dram_bw_gbs": wasted_dram_bw_gbs,
        "wasted_bw_pct": wasted_bw_pct,
        "effective_dram_bw_gbs_before": eff_bw_before,
        "effective_dram_bw_gbs_after": eff_bw_after,
        "time_saved_us": time_saved_us,
    }
