def analyze_vc_cost_tradeoff(models, trt_version_count, vc_overhead_bytes, refit_overhead_bytes):
    """Analyze storage tradeoff and breakeven version counts for version-compatible engines."""
    per_model = {}
    total_net_bytes_saved = 0
    recommended = []

    for m in sorted(models, key=lambda x: x["name"]):
        name = m["name"]
        scale = m.get("precision_scale", 1.0)
        std_bytes = int(m["base_bytes"] * scale + m["tactics_bytes"])

        refit_extra = refit_overhead_bytes if m.get("enable_refit", False) else 0
        vc_bytes = std_bytes + vc_overhead_bytes + refit_extra

        total_std = std_bytes * trt_version_count
        total_vc = vc_bytes * 1
        net_saved = total_std - total_vc

        breakeven = (vc_bytes // std_bytes) + 1 if std_bytes > 0 else 1
        is_beneficial = net_saved > 0

        per_model[name] = {
            "std_engine_bytes": std_bytes,
            "vc_engine_bytes": vc_bytes,
            "total_std_storage_bytes": total_std,
            "total_vc_storage_bytes": total_vc,
            "net_bytes_saved": net_saved,
            "breakeven_versions": breakeven,
            "is_vc_beneficial": is_beneficial,
        }

        total_net_bytes_saved += net_saved
        if is_beneficial:
            recommended.append(name)

    return {
        "per_model": per_model,
        "total_net_bytes_saved": total_net_bytes_saved,
        "recommended_vc_models": sorted(recommended),
    }
