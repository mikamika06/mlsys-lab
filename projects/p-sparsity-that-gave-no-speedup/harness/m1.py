import numpy as np

def check(workdir):
    from sparsity.core import is_2_4
    m = {"api_ok": 0.0, "true_positive": 0.0, "true_negative": 0.0, "shape_reject": 0.0}

    try:
        a = np.array([[1, 0, 0, 1], [0, 2, 0, 3]])
        b = is_2_4(a)
        m["api_ok"] = 1.0
    except Exception:
        return m

    m["true_positive"] = 1.0 if is_2_4(np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]])) else 0.0
    m["true_negative"] = 1.0 if not is_2_4(np.array([[1, 1, 1, 0]])) else 0.0

    if not is_2_4(np.array([[1, 0, 0]])):
        m["shape_reject"] = 1.0

    return m
