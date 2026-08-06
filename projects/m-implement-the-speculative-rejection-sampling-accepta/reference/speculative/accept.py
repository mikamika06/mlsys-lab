import numpy as np

def evaluate_draft(target_p, draft_p, tokens, u):
    k = target_p.shape[0]
    for i in range(k):
        p_i = target_p[i, tokens[i]]
        q_i = draft_p[i, tokens[i]]
        if u[i] >= (p_i / q_i):
            diff = np.maximum(0.0, target_p[i] - draft_p[i])
            s = np.sum(diff)
            if s > 0:
                diff /= s
            return i, diff
    return k, None
