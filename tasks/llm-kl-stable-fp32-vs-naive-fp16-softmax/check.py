import numpy as np

def _stable_fp32(logits):
    logits_f32 = logits.astype(np.float32)
    max_per_row = np.max(logits_f32, axis=1, keepdims=True)
    exp_shifted = np.exp(logits_f32 - max_per_row).astype(np.float32)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True).astype(np.float32)
    return (exp_shifted / sum_exp)

def _naive_fp16(logits):
    logits_f16 = logits.astype(np.float16)
    exp_shifted = np.exp(logits_f16).astype(np.float16)          # no stability trick
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True).astype(np.float16)
    denom_safe = np.where(sum_exp == 0,
                          np.finfo(np.float16).tiny,
                          sum_exp)
    return (exp_shifted / denom_safe)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((10, 5)).astype(np.float64)

    try:
        got = sol.kl_divergence_fp32_vs_fp16(logits)
    except Exception:
        return {"exact_match": 0.0}

    ref_stable = _stable_fp32(logits).astype(np.float64)
    ref_naive  = _naive_fp16(logits).astype(np.float64)

    p = ref_stable
    q = ref_naive

    kl_per_row = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)), axis=1)
    ref_mean_kl = float(np.mean(kl_per_row))

    if abs(got - ref_mean_kl) <= 1e-9 * max(1.0, abs(ref_mean_kl)):
        return {"exact_match": 1.0}
    else:
        return {"exact_match": 0.0}
