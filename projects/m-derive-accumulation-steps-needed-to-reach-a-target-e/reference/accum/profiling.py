def analyze_profiling_metrics(accumulation_steps, base_memory_mb, base_step_time_ms):
    peak_memory = float(base_memory_mb)
    total_wall_clock = float(base_step_time_ms * accumulation_steps)
    return {"peak_memory_mb": peak_memory, "wall_clock_ms": total_wall_clock}
