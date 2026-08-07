import numpy as np


def flash_attention_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = False,
    sm_scale: float | None = None,
):
    """Compute reference FlashAttention output [B, N, H, D] and LSE_2 [B, H, N]."""
    b, n, h, d = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / np.sqrt(d)

    q_sdpa = np.transpose(q, (0, 2, 1, 3))
    k_sdpa = np.transpose(k, (0, 2, 1, 3))
    v_sdpa = np.transpose(v, (0, 2, 1, 3))

    scores = np.matmul(q_sdpa, np.transpose(k_sdpa, (0, 1, 3, 2))) * sm_scale

    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores[:, :, mask] = -1e9

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)

    probs = exp_scores / sum_exp
    out_sdpa = np.matmul(probs, v_sdpa)

    lse_e = np.squeeze(max_scores, axis=-1) + np.log(np.squeeze(sum_exp, axis=-1))
    lse_2 = lse_e * np.log2(np.e)

    out_flash = np.ascontiguousarray(np.transpose(out_sdpa, (0, 2, 1, 3)))
    return out_flash, lse_2
