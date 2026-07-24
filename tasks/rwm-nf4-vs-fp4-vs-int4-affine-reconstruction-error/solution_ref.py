import numpy as np


def _nearest_reconstruct(x, codebook):
    idx = np.argmin(np.abs(x[:, None] - codebook[None, :]), axis=1)
    return codebook[idx]


def quantization_mse_triplet(W):
    w = np.asarray(W, dtype=np.float64)

    nf4 = np.array(
        [
            -1.0, -0.696192, -0.525073, -0.394917,
            -0.284441, -0.184773, -0.091050, 0.0,
            0.079580, 0.160930, 0.246112, 0.337915,
            0.440710, 0.562617, 0.722956, 1.0,
        ],
        dtype=np.float64,
    )
    fp4 = np.array(
        [
            -1.0, -0.66666667, -0.5, -0.33333333,
            -0.25, -0.16666667, -0.08333333, 0.0,
            0.08333333, 0.16666667, 0.25, 0.33333333,
            0.5, 0.66666667, 0.83333333, 1.0,
        ],
        dtype=np.float64,
    )

    scale = np.max(np.abs(w))
    nf4_rec = _nearest_reconstruct(w / scale, nf4) * scale
    fp4_rec = _nearest_reconstruct(w / scale, fp4) * scale

    lo = np.min(w)
    hi = np.max(w)
    int_scale = (hi - lo) / 15.0
    q = np.clip(np.round((w - lo) / int_scale), 0, 15)
    int_rec = q * int_scale + lo

    return (
        float(np.mean((w - nf4_rec) ** 2)),
        float(np.mean((w - fp4_rec) ** 2)),
        float(np.mean((w - int_rec) ** 2)),
    )
