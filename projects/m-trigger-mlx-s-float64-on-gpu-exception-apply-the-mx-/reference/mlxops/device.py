import numpy as np

class StreamContext:
    _current = "gpu"

    def __init__(self, device):
        self.device = str(device).lower()
        self._prev = None

    def __enter__(self):
        self._prev = StreamContext._current
        StreamContext._current = self.device
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        StreamContext._current = self._prev

def get_active_device():
    return StreamContext._current

def execute_op(op, tensor):
    if get_active_device() == "gpu" and getattr(tensor, "dtype", None) == np.float64:
        raise ValueError("float64 is not supported on GPU")
    return op(tensor)

def safe_float64_exec(op, tensor):
    if getattr(tensor, "dtype", None) == np.float64 and get_active_device() == "gpu":
        with StreamContext("cpu"):
            return execute_op(op, tensor)
    return execute_op(op, tensor)
