import time


class CompilationError(TypeError):
    pass


class ConstexprWrapper:

    def __init__(self, val, is_constexpr=True):
        self.val = val
        self.is_constexpr = is_constexpr

    def __int__(self):
        return int(self.val)


class MockCompiledKernel:

    def __init__(self, block_size):
        self.block_size = block_size
        time.sleep(0.01)

    def run(self, x, y):
        return [val + 1 for val in x]


class MockJITFunction:

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, x_ptr, y_ptr, BLOCK_SIZE):
        is_constexpr = getattr(BLOCK_SIZE, "is_constexpr", True)
        if not is_constexpr or isinstance(BLOCK_SIZE, list):
            raise CompilationError(
                "Parameter BLOCK_SIZE must be a tl.constexpr."
            )

        val = (
            BLOCK_SIZE.val
            if hasattr(BLOCK_SIZE, "val")
            else int(BLOCK_SIZE)
        )
        cache_key = (val,)

        if cache_key not in self.cache:
            compiled = MockCompiledKernel(val)
            self.cache[cache_key] = compiled
        else:
            compiled = self.cache[cache_key]

        return compiled.run(x_ptr, y_ptr)

    def clear_cache(self):
        self.cache.clear()


def triton_jit(fn):
    return MockJITFunction(fn)


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
