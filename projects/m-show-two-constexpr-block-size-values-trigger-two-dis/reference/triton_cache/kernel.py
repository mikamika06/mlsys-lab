import time


class CompilationError(TypeError):
    """Raised when a non-constexpr argument is passed to a constexpr parameter."""


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


@triton_jit
def mock_triton_kernel(x_ptr, y_ptr, BLOCK_SIZE):
    pass
