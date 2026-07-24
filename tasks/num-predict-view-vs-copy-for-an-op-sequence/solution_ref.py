import numpy as np


def predict_view_copy(ops):
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    answers = []
    for op in ops:
        before = arr
        if op == "reshape_2x6":
            arr = arr.reshape(2, 6)
        elif op == "slice_step2":
            arr = arr[::2]
        elif op == "transpose":
            arr = arr.T
        elif op == "ravel":
            arr = arr.ravel()
        else:
            raise ValueError(op)
        answers.append("view" if np.shares_memory(before, arr) else "copy")
    return answers
