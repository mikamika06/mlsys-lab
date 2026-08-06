import numpy as np


def ref_generate_causal_mask(sq, sk, alignment="top-left"):
    q_indices = np.arange(sq)[:, None]
    k_indices = np.arange(sk)[None, :]
    if alignment == "top-left":
        return q_indices >= (k_indices - (sk - sq))
    elif alignment == "bottom-right":
        return q_indices + (sk - sq) >= k_indices
    raise ValueError(alignment)


def ref_disagreement_map(sq, sk):
    m_tl = ref_generate_causal_mask(sq, sk, alignment="top-left")
    m_br = ref_generate_causal_mask(sq, sk, alignment="bottom-right")
    return np.not_equal(m_tl, m_br)


def ref_decode_causal_mask(sk, alignment="top-left"):
    return ref_generate_causal_mask(1, sk, alignment=alignment)


TEST_CASES = [
    (16, 16, "top-left"),
    (16, 16, "bottom-right"),
    (8, 32, "top-left"),
    (8, 32, "bottom-right"),
    (32, 8, "top-left"),
    (1, 64, "top-left"),
    (1, 64, "bottom-right"),
]
