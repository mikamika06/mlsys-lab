import ref
import numpy as np


def check(workdir):
    from memtrack.zerocopy import verify_zero_copy

    out = {"zerocopy_verified": 0.0}
    a = np.zeros(1024, dtype=np.float32)
    class DummyArray:
        def __init__(self, arr):
            self.data = self
            self.__mlx_ptr__ = arr.__array_interface__["data"][0]

    b = DummyArray(a)
    res = verify_zero_copy(a, b)
    if res is True:
        out["zerocopy_verified"] = 1.0
    else:
        out["_note"] = "zero-copy verification failed"
    return out
