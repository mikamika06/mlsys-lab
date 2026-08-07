import numpy as np


def sdpa_to_flash(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    """Convert [B, H, N, D] SDPA layout to [B, N, H, D] FlashAttention layout."""
    q_f = np.ascontiguousarray(np.transpose(q, (0, 2, 1, 3)))
    k_f = np.ascontiguousarray(np.transpose(k, (0, 2, 1, 3)))
    v_f = np.ascontiguousarray(np.transpose(v, (0, 2, 1, 3)))
    return q_f, k_f, v_f


def flash_to_sdpa(out: np.ndarray):
    """Convert [B, N, H, D] FlashAttention layout to [B, H, N, D] SDPA layout."""
    return np.ascontiguousarray(np.transpose(out, (0, 2, 1, 3)))


def lse_sdpa_to_flash(lse_e: np.ndarray) -> np.ndarray:
    """Convert base-e LSE [B, H, N] to base-2 LSE [B, H, N]."""
    return lse_e * np.log2(np.e)


def lse_flash_to_sdpa(lse_2: np.ndarray) -> np.ndarray:
    """Convert base-2 LSE [B, H, N] to base-e LSE [B, H, N]."""
    return lse_2 * np.log(2.0)
