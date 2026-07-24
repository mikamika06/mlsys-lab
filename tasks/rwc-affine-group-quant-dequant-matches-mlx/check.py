import numpy as np

def _ref_quant_dequant(weights: np.ndarray, group_size: int = 64):
    weights = np.asarray(weights, dtype=np.float64)
    n_groups = (weights.shape[0] + group_size - 1) // group_size
    q_codes = np.empty_like(weights, dtype=np.int8)
    recon = np.empty_like(weights, dtype=np.float64)
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, weights.shape[0])
        group = weights[start:end]
        bias = group.min()
        scale = (group.max() - bias) / 255.0 if group.max() != bias else 1.0
        q = np.round((group - bias) / scale)
        q_clipped = np.clip(q, -128, 127).astype(np.int8)
        recon_group = scale * q_clipped + bias
        q_codes[start:end] = q_clipped
        recon[start:end] = recon_group
    return q_codes, recon

def grade(sol, fx) -> dict:
    # Prepare deterministic test cases
    rng = np.random.default_rng(42)
    test_cases = [
        (rng.uniform(-1.0, 1.0, size=(128, 10)), 64),
        (rng.normal(size=(50, 5)), 32),
        (np.linspace(-2, 2, num=200).reshape(200, 1), 100),
    ]

    ok = 1.0
    for weights, group_size in test_cases:
        try:
            q_codes, recon = sol.affine_group_quant_dequant(weights, group_size)
        except Exception:
            return {"exact_match": 0.0}

        ref_q, ref_recon = _ref_quant_dequant(weights, group_size)

        if not np.array_equal(q_codes, ref_q):
            ok = 0.0
            break

        max_err = np.max(np.abs(recon - ref_recon))
        if max_err > 1e-6:
            ok = 0.0
            break

    return {"exact_match": ok}
