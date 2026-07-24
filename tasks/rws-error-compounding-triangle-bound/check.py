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


def _oracle(W, X, sparsity, nbits):
    def out(Wm):
        return X @ Wm.T

    denom = float(np.linalg.norm(out(W))) + 1e-12
    W_p = _prune(W, sparsity)
    W_pq = _quantize(W_p, nbits)

    e_prune = float(np.linalg.norm(out(W_p) - out(W))) / denom
    e_quant = float(np.linalg.norm(out(W_pq) - out(W_p))) / denom
    e_compound = float(np.linalg.norm(out(W_pq) - out(W))) / denom
    return e_prune, e_quant, e_compound


def grade(sol, fx) -> dict:
    """
    Builds several seeded random (W, X) trials, prunes W (zero the lowest-
    magnitude `sparsity` fraction, globally), then quantizes the pruned
    weights (per-row symmetric RTN, `nbits` bits), and computes the three
    relative output errors (through X, against the true output) with a
    NumPy oracle: prune-only, quant-only (measured on the already-pruned
    weights), and compound (quantize(prune(W)) vs. W). Compares the
    submission's three values to the oracle and checks the triangle bound
    compound <= prune + quant holds for the submission's own numbers.
    """
    rng = np.random.default_rng(0)
    rel_worst = 0.0
    bound_ok = 1.0
    for sparsity, nbits in [(0.3, 4), (0.5, 3), (0.15, 5)]:
        d_out = int(rng.integers(4, 8))
        d_in = int(rng.integers(12, 24))
        n_cal = int(rng.integers(20, 40))
        X = rng.normal(size=(n_cal, d_in))
        W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 2.0, size=(1, d_in))

        e_prune_exp, e_quant_exp, e_compound_exp = _oracle(W, X, sparsity, nbits)

        try:
            e_prune_got, e_quant_got, e_compound_got = sol.compound_error_bound(
                W.copy(), X.copy(), sparsity, nbits
            )
            e_prune_got = float(e_prune_got)
            e_quant_got = float(e_quant_got)
            e_compound_got = float(e_compound_got)
        except Exception:
            return {"errs_rel_err": float("inf"), "bound_holds": 0.0}

        for got, exp in [
            (e_prune_got, e_prune_exp),
            (e_quant_got, e_quant_exp),
            (e_compound_got, e_compound_exp),
        ]:
            rel = abs(got - exp) / (abs(exp) + 1e-12)
            rel_worst = max(rel_worst, rel)

        if e_compound_got > e_prune_got + e_quant_got + 1e-9:
            bound_ok = 0.0

    return {"errs_rel_err": rel_worst, "bound_holds": bound_ok}
