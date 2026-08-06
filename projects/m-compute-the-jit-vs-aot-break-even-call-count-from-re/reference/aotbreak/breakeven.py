import math


def compute_breakeven(profile):
    """Compute the break-even call threshold and crossover latency for a workload."""
    j_comp = profile["jit_compile_ms"]
    j_exec = profile["jit_exec_ms"]
    a_load = profile["aot_load_ms"]
    a_exec = profile["aot_exec_ms"]

    if j_exec < a_exec:
        preferred = "jit"
        setup_diff = j_comp - a_load
        exec_diff = a_exec - j_exec
    elif a_exec < j_exec:
        preferred = "aot"
        setup_diff = a_load - j_comp
        exec_diff = j_exec - a_exec
    else:
        preferred = "jit" if j_comp <= a_load else "aot"
        setup_diff = 0.0
        exec_diff = 0.0

    if exec_diff <= 0.0 or setup_diff <= 0.0:
        n_break = 1.0
    else:
        n_break = float(max(1, math.ceil(setup_diff / exec_diff)))

    if preferred == "jit":
        latency = j_comp + n_break * j_exec
    else:
        latency = a_load + n_break * a_exec

    return {
        "preferred_mode": preferred,
        "break_even_calls": float(n_break),
        "crossover_latency_ms": float(latency),
        "overhead_delta_ms": float(abs(j_comp - a_load)),
    }
