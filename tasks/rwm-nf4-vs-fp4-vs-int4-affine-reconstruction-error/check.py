import numpy as np


def _nearest_reconstruct(x, codebook):
    x = np.asarray(x, dtype=np.float64)
    idx = np.argmin(np.abs(x[:, None] - codebook[None, :]), axis=1)
    return codebook[idx]


def _nf4_mse(w):
    nf4 = np.array(
        [
            -1.000000,
            -0.696192,
            -0.525073,
            -0.394917,
            -0.284441,
            -0.184773,
            -0.091050,
            0.0,
            0.079580,
            0.160930,
            0.246112,
            0.337915,
            0.440710,
            0.562617,
            0.722956,
            1.000000,
        ],
        dtype=np.float64,
    )
    scale = np.max(np.abs(w))
    reconstructed = _nearest_reconstruct(w / scale, nf4) * scale
    return float(np.mean((w - reconstructed) ** 2))


def _fp4_mse(w):
    fp4 = np.array(
        [
            -1.0,
            -0.66666667,
            -0.5,
            -0.33333333,
            -0.25,
            -0.16666667,
            -0.08333333,
            0.0,
            0.08333333,
            0.16666667,
            0.25,
            0.33333333,
            0.5,
            0.66666667,
            0.83333333,
            1.0,
        ],
        dtype=np.float64,
    )
    scale = np.max(np.abs(w))
    reconstructed = _nearest_reconstruct(w / scale, fp4) * scale
    return float(np.mean((w - reconstructed) ** 2))


def _int4_affine_mse(w):
    lo = np.min(w)
    hi = np.max(w)
    scale = (hi - lo) / 15.0
    zero = lo
    q = np.clip(np.round((w - zero) / scale), 0, 15)
    reconstructed = q * scale + zero
    return float(np.mean((w - reconstructed) ** 2))


def _oracle(w):
    return np.array(
        [
            _nf4_mse(w),
            _fp4_mse(w),
            _int4_affine_mse(w),
        ],
        dtype=np.float64,
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(2025)
    w = rng.normal(0.0, 1.0, size=4096).astype(np.float64)

    ref = _oracle(w)

    try:
        got = np.asarray(sol.quantization_mse_triplet(w), dtype=np.float64)
    except Exception:
        return {"rel_err": 1.0}

    if got.shape != (3,):
        return {"rel_err": 1.0}

    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))

    if not (ref[0] < ref[1] < ref[2]):
        err = 1.0

    return {"rel_err": err}
