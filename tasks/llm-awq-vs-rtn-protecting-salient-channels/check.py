import numpy as np

def _ref(W, X, s):
    def quant(M):
        abs_max = np.max(np.abs(M), axis=1, keepdims=True)
        delta = abs_max / 7.0
        delta[delta == 0] = 1e-9
        M_q = np.clip(np.round(M / delta), -8, 7)
        return M_q * delta

    Y_true = X @ W.T
    Y_rtn = X @ quant(W).T
    err_rtn = np.linalg.norm(Y_rtn - Y_true) / np.linalg.norm(Y_true)

    W_scaled = W * s[np.newaxis, :]
    W_awq = quant(W_scaled) / s[np.newaxis, :]
    Y_awq = X @ W_awq.T
    err_awq = np.linalg.norm(Y_awq - Y_true) / np.linalg.norm(Y_true)

    return float(err_rtn), float(err_awq)

def grade(sol, fx) -> dict:
    cases = []
    
    np.random.seed(42)
    # Case 1
    W = np.random.randn(32, 64)
    X = np.random.randn(16, 64)
    X[:, [0, 5, 10, 15]] *= 20.0
    s = np.mean(np.abs(X), axis=0)**0.5
    cases.append((W, X, s))
    
    # Case 2
    W2 = np.random.randn(16, 128)
    X2 = np.random.randn(8, 128)
    X2[:, [10, 50, 100]] *= 50.0
    s2 = np.max(np.abs(X2), axis=0)**0.5
    cases.append((W2, X2, s2))

    awq_better = 1.0
    exact_match = 1.0
    
    for W, X, s in cases:
        ref_rtn, ref_awq = _ref(W, X, s)
        try:
            got_rtn, got_awq = sol.compare_awq_rtn(W, X, s)
        except Exception:
            return {"awq_better": 0.0, "exact_match": 0.0}
            
        if got_awq >= got_rtn:
            awq_better = 0.0
        
        if not (np.isclose(got_rtn, ref_rtn, rtol=1e-5) and np.isclose(got_awq, ref_awq, rtol=1e-5)):
            exact_match = 0.0
            
    return {"awq_better": awq_better, "exact_match": exact_match}
