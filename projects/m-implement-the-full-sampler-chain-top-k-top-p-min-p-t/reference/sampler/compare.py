import numpy as np
from sampler.chain import apply_min_p, apply_top_p


def compare_min_p_vs_top_p(logits, top_p=0.9, min_p=0.05):
    p_logits = apply_top_p(logits, top_p=top_p)
    m_logits = apply_min_p(logits, min_p=min_p)

    top_p_survivors = sorted(np.where(np.isfinite(p_logits))[0].tolist())
    min_p_survivors = sorted(np.where(np.isfinite(m_logits))[0].tolist())
    return top_p_survivors, min_p_survivors
