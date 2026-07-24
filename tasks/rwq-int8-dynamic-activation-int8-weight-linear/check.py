import numpy as np

def _oracle(X, W):
    """Reference oracle: per-channel symmetric INT8 weight, per-token symmetric INT8 activation."""
    absmax_w = np.max(np.abs(W), axis=1)
    scale_w = np.where(absmax_w > 0, absmax_w / 127.0, 1.0)
    W_q = np.clip(np.round(W / scale_w[:, np.newaxis]), -128, 127).astype(np.int8)

    absmax_x = np.max(np.abs(X), axis=1)
    scale_x = np.where(absmax_x > 0, absmax_x / 127.0, 1.0)
    X_q = np.clip(np.round(X / scale_x[:, np.newaxis]), -128, 127).astype(np.int8)

    acc = X_q.astype(np.int32) @ W_q.astype(np.int32).T
    Y = acc.astype(np.float64) * (scale_x[:, np.newaxis] * scale_w[np.newaxis, :])
    return Y

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    
    test_cases = [
        (4, 3, 5),
        (8, 7, 8),
        (16, 16, 32),
        (2, 64, 128),
    ]
    
    for B, N, K in test_cases:
        X = rng.uniform(-1.0, 1.0, size=(B, K)).astype(np.float64)
        W = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float64)
        
        Y_ref = _oracle(X, W)
        
        try:
            Y_learner = np.asarray(sol.int8_linear(X, W), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}
        
        if Y_learner.shape != (B, N):
            return {"rel_err": 1.0}
        
        denom = np.linalg.norm(Y_ref) + 1e-12
        rel_err = float(np.linalg.norm(Y_learner - Y_ref) / denom)
        
        if rel_err >= 0.001:
            return {"rel_err": rel_err}
    
    return {"rel_err": 0.0}
