import numpy as np


def _oracle_unbroadcast(grad, shape):
    """Independent oracle: pad `shape` with leading 1s to grad's rank, sum
    every axis where the padded shape is 1 in one vectorised call, then
    reshape down to `shape`. Written differently (no per-axis Python loop
    over leading dims, one combined axis-tuple reduction) from the reference
    so the two implementations don't share a bug."""
    grad = np.asarray(grad, dtype=np.float64)
    full = grad.shape
    n_extra = len(full) - len(shape)
    padded = (1,) * n_extra + tuple(shape)
    axes = tuple(i for i in range(len(full)) if padded[i] == 1 and full[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def grade(sol, fx) -> dict:
    rng = np.random.RandomState(11)
    # (broadcasted grad shape, original tensor shape)
    cases = [
        ((3, 4), (3, 4)),                 # identity, no broadcast
        ((2, 3, 4), (3, 4)),              # one extra leading dim
        ((2, 3, 4), (1, 4)),              # leading dim + interior size-1
        ((5, 2, 3, 4), (3, 4)),           # two extra leading dims
        ((5, 2, 3, 4), (1, 2, 3, 1)),     # two interior/trailing size-1 dims
        ((6,), ()),                       # broadcast scalar up to a vector
        ((1,), ()),                       # single element to true scalar
        ((4, 1, 6), (6,)),                # fewer dims + interior size-1 stretched
        ((3, 1), (1,)),                   # 2-d down to 1-d size-1
        ((2, 3, 1, 5), (3, 1, 5)),        # one extra leading dim only
        ((7, 2, 1, 4, 1), (2, 1, 4, 1)),  # multiple size-1 dims, one leading
    ]

    max_err = 0.0
    for full_shape, shape in cases:
        grad_arr = rng.randn(*full_shape)
        ref = _oracle_unbroadcast(grad_arr, shape)
        grad_list = grad_arr.tolist()
        try:
            got_list = sol.unbroadcast(grad_list, shape)
            got = np.asarray(got_list, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != tuple(shape):
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref))) if got.size else 0.0
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
