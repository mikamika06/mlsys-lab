"""Reference generator and fixture factory."""


class DummyTensor:

    def __init__(self, dtype, shape, stride):
        self.dtype = dtype
        self.shape = shape
        self.stride = stride


def get_test_cases():
    sig1 = {
        "x_ptr": {"is_ptr": True},
        "y_ptr": {"is_ptr": True},
        "n_elements": {"is_constexpr": False},
        "BLOCK_SIZE": {"is_constexpr": True},
    }

    t1 = DummyTensor("float32", (512,), (1,))
    t2 = DummyTensor("float32", (512,), (1,))
    t3 = DummyTensor("float16", (512,), (1,))

    runs1 = [
        {"x_ptr": t1, "y_ptr": t2, "n_elements": 512, "BLOCK_SIZE": 128},
        {"x_ptr": t1, "y_ptr": t2, "n_elements": 1024, "BLOCK_SIZE": 128},
        {"x_ptr": t1, "y_ptr": t3, "n_elements": 512, "BLOCK_SIZE": 128},
        {"x_ptr": t1, "y_ptr": t2, "n_elements": 512, "BLOCK_SIZE": 256},
    ]

    return [{"fn_name": "kernel_alpha", "sig": sig1, "runs": runs1}]
