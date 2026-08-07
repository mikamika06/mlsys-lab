import numpy as np

def _ref_sum_to_shape(grad, input_shape):
    """Oracle reference — uses NumPy sum to reduce along broadcasted axes."""
    grad = np.asarray(grad, dtype=np.float64)
    target_shape = grad.shape
    ndim_diff = len(target_shape) - len(input_shape)
    padded = (1,) * ndim_diff + tuple(input_shape)
    axes_to_sum = tuple(
        i for i in range(len(target_shape))
        if padded[i] == 1 and target_shape[i] > 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)
    return grad.reshape(input_shape)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    cases = [
        ((2, 3, 4), (3, 4)),          # simple leading broadcast
        ((2, 3, 4), (1, 4)),          # two leading dims, one size-1
        ((3, 4), (3, 4)),             # no broadcast (identity)
        ((5,), ()),                    # scalar input broadcast to 1-d
        ((2, 3, 4, 5), (3, 1, 5)),   # non-contiguous broadcast axes
        ((4,), (1,)),                  # 1-d size-1 input
        ((1,), ()),                    # single-element to scalar
        ((2, 1, 3), (3,)),            # fewer input dims, non-contiguous
        ((3, 1), (1,)),               # 2-d with one broadcast dim
    ]

    max_err = 0.0
    for target_shape, input_shape in cases:
        grad = rng.randn(*target_shape)
        ref = _ref_sum_to_shape(grad, input_shape)
        try:
            got = sol.sum_to_shape(grad.tolist(), input_shape)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
