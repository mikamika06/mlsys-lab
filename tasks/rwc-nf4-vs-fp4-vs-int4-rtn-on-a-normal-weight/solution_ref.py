import numpy as np

_NF4 = np.array(
    [
        -1.000000, -0.696192, -0.525073, -0.394917,
        -0.284441, -0.184773, -0.091050, 0.0,
        0.079580, 0.160930, 0.246112, 0.337915,
        0.440710, 0.562617, 0.722956, 1.000000,
    ],
    dtype=np.float64,
)

_FP4 = np.array(
    [
        -1.0, -0.66666667, -0.5, -0.33333333,
        -0.25, -0.16666667, -0.08333333, 0.0,
        0.08333333, 0.16666667, 0.25, 0.33333333,
        0.5, 0.66666667, 0.83333333, 1.0,
    ],
    dtype=np.float64,
)

_NAMES = ["NF4", "FP4", "INT4"]


def _nearest_reconstruct(x, codebook):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(len(x)):
        val = x[i]
        best_diff = None
        best_val = codebook[0]
        for c in codebook:
            diff = val - c
            if diff < 0.0:
                diff = -diff
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_val = c
        out[i] = best_val
    return out


def _codebook_mse(w, codebook):
    max_abs = 0.0
    for val in w:
        abs_val = val if val >= 0.0 else -val
        if abs_val > max_abs:
            max_abs = abs_val
    scale = max_abs
    if scale == 0.0:
        scale = 1.0

    w_scaled = np.empty_like(w)
    for i in range(len(w)):
        w_scaled[i] = w[i] / scale

    reconstructed_sub = _nearest_reconstruct(w_scaled, codebook)
    reconstructed = np.empty_like(reconstructed_sub)
    for i in range(len(reconstructed_sub)):
        reconstructed[i] = reconstructed_sub[i] * scale

    total = 0.0
    n = len(w)
    for i in range(n):
        diff = w[i] - reconstructed[i]
        total += diff * diff
    return float(total / n)


def _int4_affine_mse(w):
    lo = w[0]
    hi = w[0]
    for val in w:
        if val < lo:
            lo = val
        if val > hi:
            hi = val

    scale = (hi - lo) / 15.0
    if scale == 0.0:
        scale = 1.0
    zero = lo

    n = len(w)
    reconstructed = np.empty_like(w)
    for i in range(n):
        q = round((w[i] - zero) / scale)
        if q < 0.0:
            q = 0.0
        elif q > 15.0:
            q = 15.0
        reconstructed[i] = q * scale + zero

    total = 0.0
    for i in range(n):
        diff = w[i] - reconstructed[i]
        total += diff * diff
    return float(total / n)


def nf4_fp4_int4_best(w: np.ndarray):
    """
    w: 1-D array of weight values (assumed roughly normally distributed,
        but the function must work on whatever it's given).

    Reconstruct `w` three ways and compute each scheme's MSE:
      - NF4: absmax-scale w into [-1, 1], snap to the 16-level NF4
        codebook (quantiles of a standard normal), rescale back.
      - FP4: same absmax scaling, snapped to a 16-level FP4 codebook.
      - INT4 (RTN): a plain 4-bit AFFINE (asymmetric, min/max) uniform
        grid over 16 levels -- round-to-nearest quantization with no
        codebook shaping at all.

    Return (errors, best):
      errors: np.array([mse_nf4, mse_fp4, mse_int4]).
      best: the name of the scheme with the lowest MSE, one of
        "NF4", "FP4", "INT4".
    """
    w = np.asarray(w, dtype=np.float64)
    errs = np.array(
        [
            _codebook_mse(w, _NF4),
            _codebook_mse(w, _FP4),
            _int4_affine_mse(w),
        ],
        dtype=np.float64,
    )
    
    min_err = errs[0]
    min_idx = 0
    for i in range(1, len(errs)):
        if errs[i] < min_err:
            min_err = errs[i]
            min_idx = i

    best = _NAMES[int(min_idx)]
    return errs, best
