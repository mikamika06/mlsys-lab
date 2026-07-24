import numpy as np

def _bf16_max():
    try:
        return np.finfo(np.bfloat16).max
    except AttributeError:
        # Manual computation: (2 - 2^-7) * 2^127
        return (2.0 - 2**-7) * 2**127

def _oracle(arr):
    fp16_max = np.finfo(np.float16).max
    bf16_max = _bf16_max()
    return (np.abs(arr) > fp16_max) & (~(np.abs(arr) > bf16_max))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(seed=42)
    tests = []
    # Small array with typical values
    tests.append(rng.uniform(-1e5, 1e5, size=10))
    # Array containing values around fp16 limits
    tests.append(np.array([65504, -65505, 70000, -70000]))
    # Very large values that overflow both formats
    tests.append(np.array([1e39, -1e39, 3.5e38, -3.5e38]))
    # Array with NaNs and infinities
    tests.append(np.array([np.nan, np.inf, -np.inf, 0.0]))
    # Random large array
    tests.append(rng.standard_normal(1000) * 1e6)

    ok = 1.0
    for arr in tests:
        try:
            got = sol.which_overflow_fp16_not_bf16(arr)
            expected = _oracle(arr)
        except Exception:
            return {"exact_match": 0.0}
        if not np.array_equal(got, expected):
            ok = 0.0
            break
    return {"exact_match": ok}
