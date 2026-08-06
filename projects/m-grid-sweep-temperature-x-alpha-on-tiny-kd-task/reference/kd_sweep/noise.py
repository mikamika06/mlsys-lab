import numpy as np

def effective_temperature(teacher_logits, noise_std, T):
    rng = np.random.default_rng(42)
    noisy = teacher_logits + rng.normal(0, noise_std, size=teacher_logits.shape)
    p_orig = np.exp((teacher_logits - np.max(teacher_logits, axis=-1, keepdims=True)) / T)
    p_orig /= np.sum(p_orig, axis=-1, keepdims=True)
    p_noisy = np.exp((noisy - np.max(noisy, axis=-1, keepdims=True)) / T)
    p_noisy /= np.sum(p_noisy, axis=-1, keepdims=True)
    diff = np.mean(np.abs(p_orig - p_noisy))
    return float(diff * T)
