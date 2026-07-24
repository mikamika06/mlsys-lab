def online_softmax_stats(S):
    import numpy as np

    m = np.max(S, axis=1)
    l = np.sum(np.exp(S - m[:, None]), axis=1)

    return m.astype(np.float64), l.astype(np.float64)
