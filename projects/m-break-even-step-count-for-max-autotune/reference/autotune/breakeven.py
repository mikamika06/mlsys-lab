def compute_break_even(compile_overhead, t_default, t_autotune):
    if t_autotune >= t_default:
        return float("inf")
    return compile_overhead / (t_default - t_autotune)
