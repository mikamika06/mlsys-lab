import numpy as np


def compute_offset(n_q: int, n_kv: int, alignment: str) -> int:
    if alignment == "bottom_right":
        return n_kv - n_q
    elif alignment == "top_left":
        return 0
    else:
        raise ValueError(f"Unknown alignment: {alignment}")


def compute_causal_mask(n_q: int, n_kv: int, alignment: str = "bottom_right") -> np.ndarray:
    offset = compute_offset(n_q, n_kv, alignment)
    q_idx = np.arange(n_q)[:, None]
    kv_idx = np.arange(n_kv)[None, :]
    return kv_idx <= (q_idx + offset)
