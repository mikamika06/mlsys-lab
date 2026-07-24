import numpy as np


def _oracle(ops):
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


def grade(sol, fx) -> dict:
    cases = [
        ["reshape_2x6", "ravel"],
        ["transpose", "ravel"],
        ["slice_step2", "ravel"],
        ["transpose", "reshape_2x6", "ravel"],
        ["slice_step2", "transpose", "ravel"],
        ["reshape_2x6", "transpose", "ravel"],
    ]
    ok = 1.0
    for ops in cases:
        try:
            got = sol.predict_view_copy(list(ops))
        except Exception:
            ok = 0.0
            break
        expected = _oracle(ops)
        if list(got) != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
