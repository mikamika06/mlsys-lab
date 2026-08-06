def pick_mode(compile_overhead, t_default, t_reduce_overhead, t_max_autotune, steps):
    times = {
        "default": t_default * steps,
        "reduce-overhead": compile_overhead + t_reduce_overhead * steps,
        "max-autotune": compile_overhead + t_max_autotune * steps,
    }
    return min(times, key=times.get)
