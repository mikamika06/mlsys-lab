import numpy as np
from famask.generator import generate_causal_mask


def disagreement_map(sq, sk):
    m_tl = generate_causal_mask(sq, sk, alignment="top-left")
    m_br = generate_causal_mask(sq, sk, alignment="bottom-right")
    return np.not_equal(m_tl, m_br)
