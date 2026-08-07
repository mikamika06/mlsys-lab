def mock_triton_kernel(x_ptr, y_ptr, BLOCK_SIZE):
    raise NotImplementedError


class MockCompiledKernel:

    def __init__(self, block_size):
        raise NotImplementedError


class MockJITFunction:

    def __init__(self, fn):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def clear_cache(self):
        raise NotImplementedError


def triton_jit(fn):
    raise NotImplementedError
