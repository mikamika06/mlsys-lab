def predict_startup_time(phase_times, warm_compile_cache, cache_speedup_factor):
    """Predict second-start startup time given warm compile cache."""
    raise NotImplementedError


def compute_safe_cooldown_period(phase_times, warm_compile_cache, cache_speedup_factor, safety_margin_pct):
    """Compute safe HPA cooldownPeriod in seconds."""
    raise NotImplementedError
