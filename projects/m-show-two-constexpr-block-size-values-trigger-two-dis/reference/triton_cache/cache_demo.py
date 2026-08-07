import time
from triton_cache.kernel import CompilationError


class ConstexprWrapper:

    def __init__(self, val, is_constexpr=True):
        self.val = val
        self.is_constexpr = is_constexpr

    def __int__(self):
        return int(self.val)


def inspect_cache_keys(kernel_fn, block_size_a, block_size_b):
    kernel_fn.clear_cache()
    c_a = ConstexprWrapper(block_size_a, is_constexpr=True)
    c_b = ConstexprWrapper(block_size_b, is_constexpr=True)

    kernel_fn([1, 2], [0, 0], c_a)
    kernel_fn([1, 2], [0, 0], c_b)

    return list(kernel_fn.cache.keys())


def trigger_constexpr_error(kernel_fn, non_constexpr_val):
    c_invalid = ConstexprWrapper(non_constexpr_val, is_constexpr=False)
    try:
        kernel_fn([1, 2], [0, 0], c_invalid)
    except CompilationError as e:
        return True, str(e)
    except Exception as e:
        return False, f"Unexpected error type: {type(e).__name__}"
    return False, "No exception raised"


def measure_compile_vs_hit_latency(kernel_fn, block_size):
    kernel_fn.clear_cache()
    c_val = ConstexprWrapper(block_size, is_constexpr=True)

    t0 = time.perf_counter()
    kernel_fn([1, 2, 3, 4], [0, 0, 0, 0], c_val)
    t1 = time.perf_counter()
    cold_latency = t1 - t0

    t2 = time.perf_counter()
    kernel_fn([1, 2, 3, 4], [0, 0, 0, 0], c_val)
    t3 = time.perf_counter()
    warm_latency = t3 - t2

    return cold_latency, warm_latency
