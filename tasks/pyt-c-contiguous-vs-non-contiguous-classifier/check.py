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


def _oracle(ops):
    arr = np.arange(24, dtype=np.int64).reshape(4, 6)
    answers = []
    for op in ops:
        arr = _apply(op, arr)
        answers.append(bool(memoryview(arr).c_contiguous))
    return answers


def grade(sol, fx) -> dict:
    cases = [
        ["transpose"],
        ["slice_step2"],
        ["reshape_flat"],
        ["flip"],
        ["transpose", "reshape_flat"],
        ["slice_step2", "reshape_flat"],
        ["flip", "reshape_flat"],
        ["transpose", "transpose"],
        ["slice_step2", "transpose"],
        ["transpose", "flip"],
    ]
    ok = 1.0
    for ops in cases:
        try:
            got = sol.predict_c_contiguous(list(ops))
        except Exception:
            ok = 0.0
            break
        expected = _oracle(ops)
        if [bool(g) for g in got] != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
