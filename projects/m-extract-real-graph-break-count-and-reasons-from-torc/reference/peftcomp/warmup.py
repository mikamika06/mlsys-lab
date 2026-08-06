import time


def measure_warmup_cost(compiled_fn, sample_inputs):
    if not isinstance(sample_inputs, (list, tuple)):
        sample_inputs = (sample_inputs,)

    start_compile = time.perf_counter_ns()
    _ = compiled_fn(*sample_inputs)
    end_compile = time.perf_counter_ns()
    compile_duration = end_compile - start_compile

    start_cached = time.perf_counter_ns()
    _ = compiled_fn(*sample_inputs)
    end_cached = time.perf_counter_ns()
    cached_duration = end_cached - start_cached

    gap = max(0, compile_duration - cached_duration)
    return {
        "compile_ns": compile_duration,
        "cached_ns": cached_duration,
        "warmup_gap_ns": gap
    }
