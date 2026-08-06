import ref
import numpy as np

def check(workdir):
    from speculative.sampling import modified_rejection_sampling
    p_d = [0.2, 0.3, 0.5]
    p_t = [0.1, 0.4, 0.5]
    seed = 123
    got = modified_rejection_sampling(p_d, p_t, seed)
    want = ref.modified_rejection_sampling(p_d, p_t, seed)
    match = float(np.all(np.array(got) == np.array(want)))
    return {"lossless_score": match}
