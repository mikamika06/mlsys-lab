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
    idx = np.argmin(np.abs(x[:, None] - codebook[None, :]), axis=1)
    return codebook[idx]


def _codebook_mse(w, codebook):
    scale = np.max(np.abs(w))
    if scale == 0:
        scale = 1.0
    reconstructed = _nearest_reconstruct(w / scale, codebook) * scale
    return float(np.mean((w - reconstructed) ** 2))


def _int4_affine_mse(w):
    lo = np.min(w)
    hi = np.max(w)
    scale = (hi - lo) / 15.0
    if scale == 0:
        scale = 1.0
    zero = lo
    q = np.clip(np.round((w - zero) / scale), 0, 15)
    reconstructed = q * scale + zero
    return float(np.mean((w - reconstructed) ** 2))


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
    best = _NAMES[int(np.argmin(errs))]
    return errs, best
