import numpy as np

def measure_sdpa_scale_and_top_logit(
    Q: np.ndarray,
    K: np.ndarray,
    *,
    scale: float | None = None
) -> tuple[float, float]:
    head_dim = Q.shape[-1]
    used_scale = 1 / np.sqrt(head_dim) if scale is None else float(scale)
    logits = np.matmul(Q, K.transpose(0, 1, 3, 2)) * used_scale
    top_logit = np.max(logits)
    return used_scale, float(top_logit)
