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


def _oracle(w):
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


def grade(sol, fx) -> dict:
    cases = []
    rng1 = np.random.default_rng(2025)
    cases.append(rng1.normal(0.0, 1.0, size=4096))
    rng2 = np.random.default_rng(7)
    cases.append(rng2.normal(0.0, 2.5, size=1500))
    rng3 = np.random.default_rng(99)
    cases.append(rng3.normal(0.0, 0.01, size=800))
    # a case that should NOT favor NF4: a heavy-tailed / bimodal weight,
    # to make sure the "best" answer genuinely varies with the data.
    rng4 = np.random.default_rng(13)
    half = rng4.normal(-3.0, 0.2, size=1000)
    other = rng4.normal(3.0, 0.2, size=1000)
    cases.append(np.concatenate([half, other]))

    worst = 0.0
    for w in cases:
        ref_errs, ref_best = _oracle(w)
        try:
            got_errs, got_best = sol.nf4_fp4_int4_best(w.copy())
            got_errs = np.asarray(got_errs, dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        if got_errs.shape != (3,):
            return {"rel_err": 1.0}

        err = float(np.linalg.norm(got_errs - ref_errs) / (np.linalg.norm(ref_errs) + 1e-12))
        if str(got_best) != ref_best:
            err = 1.0
        worst = max(worst, err)

    return {"rel_err": worst}
