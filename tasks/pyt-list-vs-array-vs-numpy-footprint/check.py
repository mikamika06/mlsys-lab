import sys
from array import array


def _ref(n: int) -> dict[str, float]:
    values = list(range(n))
    list_bytes = sys.getsizeof(values)

    arr = array("q", values)
    array_bytes = sys.getsizeof(arr)

    return {
        "list_vs_array": list_bytes / array_bytes,
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
