import numpy as np


def cascade_stage1_accept(q1: np.ndarray, q2: np.ndarray, x1: int, rng: np.random.Generator) -> tuple[bool, int]:
    """Accept or resample token x1 proposed by q1 for stage 2 distribution q2."""
    prob = min(1.0, float(q2[x1] / q1[x1]))
    if rng.uniform() < prob:
        return True, int(x1)
    res = np.maximum(0.0, q2 - q1)
    s = np.sum(res)
    if s < 1e-12:
        p_res = q2 / np.sum(q2)
    else:
        p_res = res / s
    x2 = int(rng.choice(len(q2), p=p_res))
    return False, x2


def cascade_stage2_accept(q2: np.ndarray, p: np.ndarray, x2: int, rng: np.random.Generator) -> tuple[bool, int]:
    """Accept or resample token x2 from stage 2 for target distribution p."""
    prob = min(1.0, float(p[x2] / q2[x2]))
    if rng.uniform() < prob:
        return True, int(x2)
    res = np.maximum(0.0, p - q2)
    s = np.sum(res)
    if s < 1e-12:
        p_res = p / np.sum(p)
    else:
        p_res = res / s
    x_final = int(rng.choice(len(p), p=p_res))
    return False, x_final


def multi_draft_select(candidates: list[int], q_drafts: list[np.ndarray], p: np.ndarray, rng: np.random.Generator) -> tuple[bool, int, int]:
    """Select token from K independent draft proposals using optimal mixture acceptance."""
    k_drafts = len(q_drafts)
    q_mix = np.mean(q_drafts, axis=0)
    idx = int(rng.integers(0, k_drafts))
    cand_tok = candidates[idx]
    prob = min(1.0, float(p[cand_tok] / q_mix[cand_tok]))
    if rng.uniform() < prob:
        return True, idx, int(cand_tok)
    res = np.maximum(0.0, p - q_mix)
    s = np.sum(res)
    if s < 1e-12:
        p_res = p / np.sum(p)
    else:
        p_res = res / s
    x_final = int(rng.choice(len(p), p=p_res))
    return False, -1, x_final
