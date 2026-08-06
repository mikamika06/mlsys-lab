"""Wall-clock timing comparison tools."""

import time

def profile_runtimes(pt_step_fn, mlx_step_fn, steps, warmup_steps=2):
    for _ in range(warmup_steps):
        pt_step_fn()
    pt_start = time.perf_counter()
    for _ in range(steps):
        pt_step_fn()
    pt_total = time.perf_counter() - pt_start

    for _ in range(warmup_steps):
        mlx_step_fn()
    mlx_start = time.perf_counter()
    for _ in range(steps):
        mlx_step_fn()
    mlx_total = time.perf_counter() - mlx_start

    ratio = pt_total / mlx_total if mlx_total > 0 else 0.0
    return {
        "pt_total_sec": pt_total,
        "mlx_total_sec": mlx_total,
        "latency_ratio": ratio,
        "pt_avg_step_sec": pt_total / steps if steps > 0 else 0.0,
        "mlx_avg_step_sec": mlx_total / steps if steps > 0 else 0.0
    }
