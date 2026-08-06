"""Version-compatible engine size cost model."""


def estimate_vc_engine_cost(base_engine_mb, enable_vc, refit_enabled, lean_runtime):
    size = float(base_engine_mb)
    if enable_vc:
        size *= 1.35
    if refit_enabled:
        size *= 1.15
    if lean_runtime:
        size *= 0.85
    size = round(size, 4)
    delta_mb = round(size - base_engine_mb, 4)
    return {
        "final_size_mb": size,
        "delta_mb": delta_mb,
        "ratio": round(size / base_engine_mb, 4),
    }
