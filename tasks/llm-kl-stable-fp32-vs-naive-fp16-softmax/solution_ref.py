import numpy as np

def kl_divergence_fp32_vs_fp16(logits: np.ndarray) -> float:
    """
    Compute the mean KL divergence between a stable FP32 softmax and a naïve FP16 softmax.
    """
    # Stable FP32 softmax
    logits_f32 = logits.astype(np.float32)
    max_per_row = np.max(logits_f32, axis=1, keepdims=True)
    exp_shifted = np.exp(logits_f32 - max_per_row).astype(np.float32)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True).astype(np.float32)
    softmax_fp32 = (exp_shifted / sum_exp)

    # Naïve FP16 softmax
    logits_f16 = logits.astype(np.float16)
    exp_shifted_f16 = np.exp(logits_f16).astype(np.float16)          # no stability trick
    sum_exp_f16 = np.sum(exp_shifted_f16, axis=1, keepdims=True).astype(np.float16)
    denom_safe_f16 = np.where(sum_exp_f16 == 0,
                              np.finfo(np.float16).tiny,
                              sum_exp_f16)
    softmax_fp16 = (exp_shifted_f16 / denom_safe_f16)

    # Convert to float64 for KL computation
    p = softmax_fp32.astype(np.float64)
    q = softmax_fp16.astype(np.float64)

    # Mean KL divergence over rows
    kl_per_row = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)), axis=1)
    return float(np.mean(kl_per_row))
