import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    n_cases = 10
    batch = rng.integers(1, 64, size=n_cases)
    seq   = rng.integers(1, 2048, size=n_cases)
    d     = rng.choice([16, 32, 64, 128, 256, 512, 1024, 2048], size=n_cases)
    dtype_choices = ['float16', 'float32']
    dtype = np.array(rng.choice(dtype_choices, size=n_cases))
    peak_flops = rng.uniform(100e12, 300e12, size=n_cases)   # FLOPs/s
    peak_bw    = rng.uniform(500e9, 2000e9, size=n_cases)     # Bytes/s

    try:
        got = sol.predict_regime(batch, seq, d, dtype, peak_flops, peak_bw)
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(got, np.ndarray):
        return {"exact_match": 0.0}

    # Reference calculation
    bpe = np.array([np.dtype(dt).itemsize for dt in dtype], dtype=np.int64)
    flops_total = 2 * d**2 * seq
    memory_bytes = (d*d + 2*d) * bpe
    oi = flops_total / memory_bytes
    ridge = peak_flops / peak_bw
    ref = (oi > ridge).astype(int)

    # Add a deterministic case that forces a mismatch if the starter is wrong
    batch_det = np.array([8])
    seq_det   = np.array([1024])
    d_det     = np.array([256])
    dtype_det = np.array(['float32'])
    peak_flops_det = np.array([200e12])  # ridge = 300
    peak_bw_det    = np.array([666e9])

    try:
        got_det = sol.predict_regime(batch_det, seq_det, d_det, dtype_det,
                                     peak_flops_det, peak_bw_det)
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(got_det, np.ndarray):
        return {"exact_match": 0.0}

    bpe_det = np.array([np.dtype(dt).itemsize for dt in dtype_det], dtype=np.int64)
    flops_total_det = 2 * d_det**2 * seq_det
    memory_bytes_det = (d_det*d_det + 2*d_det) * bpe_det
    oi_det = flops_total_det / memory_bytes_det
    ridge_det = peak_flops_det / peak_bw_det
    ref_det = (oi_det > ridge_det).astype(int)

    ok = int(np.array_equal(got, ref) and np.array_equal(got_det, ref_det))
    return {"exact_match": float(ok)}
