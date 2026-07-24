import numpy as np

from mlsys.scorers import byte_exact_fraction


def _script():
    shape = (3, 4)
    dtype = np.float32
    rng = np.random.default_rng(0)

    a1 = rng.standard_normal(shape).astype(dtype)
    a2 = rng.standard_normal(shape).astype(dtype)
    a3 = rng.standard_normal(shape).astype(dtype)
    a4 = rng.standard_normal(shape).astype(dtype)

    wrong_shape = rng.standard_normal((4, 3)).astype(dtype)
    wrong_dtype_f64 = rng.standard_normal(shape).astype(np.float64)
    wrong_dtype_i32 = rng.standard_normal(shape).astype(np.int32)
    wrong_shape2 = rng.standard_normal((2, 4)).astype(dtype)
    not_array = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    # (value_to_assign, expected_valid, expected_buffer_after)
    ops = []
    last_good = None

    def valid(a):
        nonlocal last_good
        last_good = a
        ops.append((a, True, a))

    def invalid(v):
        ops.append((v, False, last_good))

    valid(a1)
    invalid(wrong_shape)
    invalid(wrong_dtype_f64)
    valid(a2)
    invalid(not_array)
    valid(a3)
    invalid(wrong_dtype_i32)
    valid(a4)
    invalid(wrong_shape2)
    invalid("not an array at all")

    return shape, dtype, ops


def grade(sol, fx) -> dict:
    shape, dtype, ops = _script()

    try:
        va = sol.ValidatedArray(shape, dtype=dtype)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    for value, expected_valid, _expected_buf in ops:
        try:
            va.data = value
            accepted = True
        except Exception:
            accepted = False
        if accepted != expected_valid:
            return {"byte_exact_fraction": 0.0}

    expected_final = ops[-1][2]

    try:
        got = np.asarray(va.data)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    if got.shape != expected_final.shape or got.dtype != expected_final.dtype:
        return {"byte_exact_fraction": 0.0}

    return {"byte_exact_fraction": byte_exact_fraction(expected_final, got)}
