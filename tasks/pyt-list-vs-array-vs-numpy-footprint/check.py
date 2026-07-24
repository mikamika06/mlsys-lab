import sys
from array import array
import numpy as np


def _ref(n):
    values = list(range(n))
    list_bytes = sys.getsizeof(values)

    arr = array("q", values)
    array_bytes = sys.getsizeof(arr)

    np_arr = np.asarray(values, dtype=np.int64)
    numpy_bytes = np_arr.nbytes

    return {
        "list_vs_array": list_bytes / array_bytes,
        "list_vs_numpy": list_bytes / numpy_bytes,
        "array_vs_numpy": array_bytes / numpy_bytes,
    }


def grade(sol, fx) -> dict:
    cases = [1, 2, 8, 64, 1000]
    max_err = 0.0

    for n in cases:
        try:
            got = sol.footprint_ratios(n)
        except Exception:
            return {"max_abs_err": 1.0}

        ref = _ref(n)
        for key, value in ref.items():
            try:
                err = abs(float(got[key]) - value)
            except Exception:
                return {"max_abs_err": 1.0}
            max_err = max(max_err, err)

    return {"max_abs_err": max_err}
