import numpy as np


def _prune(W, sparsity):
    Wf = W.ravel().copy()
    n_prune = int(round(sparsity * Wf.size))
    order = np.argsort(np.abs(Wf), kind="stable")
    Wf[order[:n_prune]] = 0.0
    return Wf.reshape(W.shape)


def _quantize(W, nbits):
    qmax = (1 << (nbits - 1)) - 1
    amax = np.max(np.abs(W), axis=1)
    s = np.where(amax > 0, amax / qmax, 1.0)
    codes = np.clip(np.round(W / s[:, None]), -qmax, qmax)
    return codes * s[:, None]


def compound_error_bound(W: np.ndarray, X: np.ndarray, sparsity: float, nbits: int):
    """
    1. Prune W: zero the lowest-magnitude `sparsity` fraction of entries,
       globally, by stable-sorted |W| rank -> W_p.
    2. Quantize the PRUNED weights: per-row symmetric RTN at `nbits` bits
       -> W_pq.
    3. Output errors, relative to ||X W^T|| (through the linear layer):
       e_prune    = ||X W_p^T  - X W^T ||  / ||X W^T||
       e_quant    = ||X W_pq^T - X W_p^T|| / ||X W^T||   (quant error on the pruned weights)
       e_compound = ||X W_pq^T - X W^T ||  / ||X W^T||

    Returns (e_prune, e_quant, e_compound). Because
    (W_pq - W) = (W_pq - W_p) + (W_p - W) exactly, the triangle
    inequality on the Frobenius norm guarantees e_compound <= e_prune + e_quant.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    def out(Wm):
        return X @ Wm.T

    denom = float(np.linalg.norm(out(W))) + 1e-12
    W_p = _prune(W, sparsity)
    W_pq = _quantize(W_p, nbits)

    e_prune = float(np.linalg.norm(out(W_p) - out(W))) / denom
    e_quant = float(np.linalg.norm(out(W_pq) - out(W_p))) / denom
    e_compound = float(np.linalg.norm(out(W_pq) - out(W))) / denom
    return e_prune, e_quant, e_compound
