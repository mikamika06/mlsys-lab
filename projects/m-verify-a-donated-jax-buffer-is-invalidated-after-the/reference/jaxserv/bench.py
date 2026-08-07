import math

def measure_peak_memory_savings(state_shape, dtype_bytes, num_updates):
    size_bytes = 1
    for dim in state_shape:
        size_bytes *= dim
    size_bytes *= dtype_bytes
    without_donation_peak = size_bytes * 2
    with_donation_peak = size_bytes
    bytes_saved = without_donation_peak - with_donation_peak
    saved_ratio = bytes_saved / without_donation_peak
    return {
        "without_donation_peak": without_donation_peak,
        "with_donation_peak": with_donation_peak,
        "bytes_saved": bytes_saved,
        "saved_ratio": saved_ratio,
    }


def compute_breakeven_requests(compile_time_ms, eager_step_ms, compiled_step_ms) -> int:
    if compiled_step_ms >= eager_step_ms:
        return -1
    diff = eager_step_ms - compiled_step_ms
    return math.ceil(compile_time_ms / diff)
