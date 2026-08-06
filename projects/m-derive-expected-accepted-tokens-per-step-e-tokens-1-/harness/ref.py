import numpy as np

def expected_tokens(a, g):
    if abs(a - 1.0) < 1e-9:
        return float(g + 1)
    return float((1.0 - a**(g + 1)) / (1.0 - a))

def modified_rejection_sampling(p_draft, p_target, seed):
    rng = np.random.default_rng(seed)
    pd = np.array(p_draft, dtype=float)
    pt = np.array(p_target, dtype=float)
    pd /= pd.sum()
    pt /= pt.sum()
    r = rng.random(len(pd))
    accepted = r < np.minimum(1.0, pt / pd)
    return accepted

def generate_trace(p_draft, p_target, seed, steps):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        pd = np.array(p_draft, dtype=float)
        pt = np.array(p_target, dtype=float)
        pd /= pd.sum()
        pt /= pt.sum()
        r = rng.random(len(pd))
        acc = r < np.minimum(1.0, pt / pd)
        out.append(acc.tolist())
    return out
