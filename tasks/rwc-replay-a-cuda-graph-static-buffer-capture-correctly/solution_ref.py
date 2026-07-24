import numpy as np


def static_buffer_replay(W):
    W = np.asarray(W, dtype=np.float64)
    static_in = None
    static_out = None

    def replay(X):
        nonlocal static_in, static_out
        X = np.asarray(X, dtype=np.float64)
        if static_in is None:
            static_in = np.empty_like(X)
            static_out = np.empty((X.shape[0], W.shape[0]), dtype=np.float64)

        np.copyto(static_in, X)
        np.copyto(static_out, static_in @ W.T)
        return static_out.copy()

    return replay
