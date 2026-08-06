import time

def compare_compilation_timings(jit_fn, *args):
    """Profile AOT compilation time vs JIT first-call and cached execution times."""
    target_aot = jit_fn.fresh_instance() if hasattr(jit_fn, "fresh_instance") else jit_fn
    lowered = target_aot.lower(*args)

    t0 = time.perf_counter()
    _ = lowered.compile()
    t1 = time.perf_counter()
    aot_compile_time = t1 - t0

    target_jit = jit_fn.fresh_instance() if hasattr(jit_fn, "fresh_instance") else jit_fn
    t2 = time.perf_counter()
    _ = target_jit(*args)
    t3 = time.perf_counter()
    jit_first_call_time = t3 - t2

    t4 = time.perf_counter()
    _ = target_jit(*args)
    t5 = time.perf_counter()
    jit_cached_time = t5 - t4

    denom = max(jit_first_call_time - jit_cached_time, 1e-9)
    overhead_ratio = aot_compile_time / denom

    return {
        "aot_compile_time": float(aot_compile_time),
        "jit_first_call_time": float(jit_first_call_time),
        "jit_cached_time": float(jit_cached_time),
        "overhead_ratio": float(overhead_ratio),
    }
