import numpy as np


def _oracle(Q, K, V, seq_lens):
    """Per-sequence attention, computed by slicing out each sequence and
    running ordinary dense attention on it alone -- never referencing a
    block-diagonal mask, so it is an independent check on the packed
    implementation."""
    d = Q.shape[1]
    outs = []
    pos = 0
    for L in seq_lens:
        Qs = Q[pos:pos + L]
        Ks = K[pos:pos + L]
        Vs = V[pos:pos + L]
        scores = (Qs @ Ks.T) / np.sqrt(d)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        outs.append(probs @ Vs)
        pos += L
    return np.concatenate(outs, axis=0)


def _cases():
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(6):
        n_seqs = int(rng.integers(1, 5))
        lens = [int(rng.integers(1, 9)) for _ in range(n_seqs)]
        N = sum(lens)
        d = int(rng.integers(2, 8))
        Q = rng.standard_normal((N, d))
        K = rng.standard_normal((N, d))
        V = rng.standard_normal((N, d))
        cases.append((Q, K, V, lens))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for Q, K, V, seq_lens in _cases():
        ref = _oracle(Q, K, V, seq_lens)
        try:
            got = np.asarray(sol.block_diagonal_attention(Q, K, V, list(seq_lens)), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
