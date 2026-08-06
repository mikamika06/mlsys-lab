import numpy as np

def modified_rejection_sampling(p_draft, p_target, seed):
    rng = np.random.default_rng(seed)
    pd = np.array(p_draft, dtype=float)
    pt = np.array(p_target, dtype=float)
    pd /= pd.sum()
    pt /= pt.sum()
    r = rng.random(len(pd))
    accepted = r < np.minimum(1.0, pt / pd)
    return accepted
