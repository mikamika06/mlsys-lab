import numpy as np

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
