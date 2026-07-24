import numpy as np

BLOCK_SIZE = 32
FP4_MAX = 6.0  # max representable magnitude of the MXFP4 (E2M1) element format


def _oracle(x: np.ndarray, block_size: int) -> np.ndarray:
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)
    amax = np.max(np.abs(xb), axis=1)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(amax / FP4_MAX))
    exp = np.where(amax == 0, 0.0, exp)
    return exp.astype(np.int64)


def grade(sol, fx) -> dict:
    x = fx["mx_w"]
    ref = _oracle(x, BLOCK_SIZE)

    try:
        got = sol.mxfp4_block_exponent(x.copy(), BLOCK_SIZE)
        got = np.asarray(got).astype(np.int64).reshape(-1)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != ref.shape:
        return {"exact_match": 0.0}

    return {"exact_match": float(np.array_equal(got, ref))}
