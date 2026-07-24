import numpy as np

from mlsys import scorers


def _decode_fn(patch: np.ndarray) -> np.ndarray:
    """Stand-in for a VAE decode: a fixed 5x5 local (`valid`-mode) box
    blur + nonlinearity. Has local support only -- shrinks the array by
    2 on every side (a 5-wide window has radius 2). This is the
    `decode_fn` passed into `tiled_decode`; it never sees anything
    outside the array it's given."""
    p = np.asarray(patch, dtype=np.float64)
    k = 5
    if p.shape[0] < k or p.shape[1] < k:
        raise ValueError("patch too small for decode_fn")
    windows = np.lib.stride_tricks.sliding_window_view(p, (k, k))
    out = windows.mean(axis=(-1, -2))
    return np.tanh(out)


def _full_decode(image: np.ndarray, r: int = 2) -> np.ndarray:
    """The untiled ground truth: reflect-pad the WHOLE image once at its
    true boundary, then decode in one shot."""
    padded = np.pad(np.asarray(image, dtype=np.float64), r, mode="reflect")
    return _decode_fn(padded)


def _build_cases():
    cases = [(24, 24, 8, 4), (25, 22, 7, 3), (16, 16, 4, 2)]
    out = []
    for seed, (H, W, tile_size, overlap) in enumerate(cases):
        rng = np.random.default_rng(seed)
        image = rng.normal(size=(H, W))
        out.append((image, tile_size, overlap))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for image, tile_size, overlap in _build_cases():
        ref = _full_decode(image)
        try:
            got = sol.tiled_decode(image.copy(), _decode_fn, tile_size, overlap)
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(ref, got)
        if not np.isfinite(err):
            err = float("inf")
        worst = max(worst, err)

    return {"max_abs_err": worst}
