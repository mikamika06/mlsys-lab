def identify_salient_channels(X):
    import numpy as np
    axes = tuple(i for i in range(X.ndim) if i != 1)
    s = np.mean(np.abs(X), axis=axes)
    k = int(np.ceil(s.size * 0.01))
    top = np.argsort(-s)[:k]
    return sorted(int(idx) for idx in top)
