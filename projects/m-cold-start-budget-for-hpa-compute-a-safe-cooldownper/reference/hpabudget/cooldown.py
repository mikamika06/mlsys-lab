import math


def predict_startup_time(phase_times, warm_compile_cache, cache_speedup_factor):
    """Predict second-start startup time given warm compile cache."""
    compile_time = phase_times.get("torch_compile", 0.0)
    if warm_compile_cache:
        compile_time = compile_time / cache_speedup_factor

    total = (
        phase_times.get("process_bootstrap", 0.0)
        + phase_times.get("weight_loading", 0.0)
        + compile_time
        + phase_times.get("cudagraph_capture", 0.0)
    )
    return total


def compute_safe_cooldown_period(phase_times, warm_compile_cache, cache_speedup_factor, safety_margin_pct):
    """Compute safe HPA cooldownPeriod in seconds."""
    predicted_start = predict_startup_time(phase_times, warm_compile_cache, cache_speedup_factor)
    multiplier = 1.0 + (safety_margin_pct / 100.0)
    safe_period = math.ceil(predicted_start * multiplier)
    return int(safe_period)
