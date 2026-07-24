import numpy as np


def _apply(op, arr):
    if op == "transpose":
        return arr.T
    if op == "slice_step2":
        return arr[::2]
    if op == "reshape_flat":
        return arr.reshape(-1)
    if op == "flip":
        return arr[::-1]
    raise ValueError(op)


def predict_c_contiguous(ops):
    """Replay ops from the base array, reporting buffer-protocol C-contiguity after each."""
    arr = np.arange(24, dtype=np.int64).reshape(4, 6)
    answers = []
    for op in ops:
        arr = _apply(op, arr)
        answers.append(bool(memoryview(arr).c_contiguous))
    return answers
