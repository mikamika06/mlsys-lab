import numpy as np


class StaticBufferHarness:
    """Static buffer harness managing variable input sizes for captured graph executions."""

    def __init__(self, max_shape, dtype=np.float32):
        self.max_shape = tuple(max_shape)
        self.dtype = dtype
        self.buffer = np.zeros(self.max_shape, dtype=self.dtype)
        self.active_slice = None

    def update_input(self, tensor):
        tensor = np.asarray(tensor, dtype=self.dtype)
        if len(tensor.shape) != len(self.max_shape):
            raise ValueError("Dimensions mismatch")
        for s_in, s_max in zip(tensor.shape, self.max_shape):
            if s_in > s_max:
                raise ValueError("Input tensor exceeds static buffer size")

        self.buffer.fill(0)
        slices = tuple(slice(0, s) for s in tensor.shape)
        self.buffer[slices] = tensor
        self.active_slice = slices
        return self.buffer

    def run(self, graph_runner):
        if self.active_slice is None:
            raise RuntimeError("No input set in harness")
        full_out = graph_runner(self.buffer)
        return full_out[self.active_slice]
