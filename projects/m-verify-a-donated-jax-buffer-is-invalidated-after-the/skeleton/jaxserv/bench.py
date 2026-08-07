def measure_peak_memory_savings(state_shape, dtype_bytes, num_updates):
    raise NotImplementedError


def compute_breakeven_requests(compile_time_ms, eager_step_ms, compiled_step_ms) -> int:
    raise NotImplementedError
