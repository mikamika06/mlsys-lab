"""Grade prefill-vs-decode FLOP accounting against a NumPy execution oracle.

The oracle never hardcodes the closed form. It builds the real layer weights,
runs an actual prefill forward pass and an actual decode forward pass in NumPy,
and tallies 2*m*k*n for every matmul it executes (reading the shapes off the
real operands it feeds to ``@``). That execution-derived count is the reference.
"""
import numpy as np


def _oracle_flops(d_model, n_heads, d_ff, P, T, seed=0):
    """Run a real one-layer forward pass and count matmul FLOPs by execution."""
    rng = np.random.default_rng(seed)
    dh = d_model // n_heads
    flops = [0]

    def mm(a, b):
        # a: (m, k), b: (k, n)  ->  2 * m * k * n FLOPs, counted from real shapes
        flops[0] += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b

    Wq = rng.standard_normal((d_model, d_model))
    Wk = rng.standard_normal((d_model, d_model))
    Wv = rng.standard_normal((d_model, d_model))
    Wo = rng.standard_normal((d_model, d_model))
    Wup = rng.standard_normal((d_model, d_ff))
    Wdn = rng.standard_normal((d_ff, d_model))

    # ---- PREFILL: P tokens, no cache, self-attention over P positions ----
    X = rng.standard_normal((P, d_model))
    Q = mm(X, Wq)
    K = mm(X, Wk)
    V = mm(X, Wv)
    ctx = np.empty((P, d_model))
    for h in range(n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        scores = mm(Q[:, sl], K[:, sl].T)      # (P, P)
        ctx[:, sl] = mm(scores, V[:, sl])      # (P, dh)
    O = mm(ctx, Wo)
    Hup = mm(O, Wup)
    mm(Hup, Wdn)
    prefill = flops[0]

    # ---- DECODE: 1 new token attending to T cached keys/values ----
    flops[0] = 0
    x = rng.standard_normal((1, d_model))
    q = mm(x, Wq)
    mm(x, Wk)   # new-token K projection (appended to cache)
    mm(x, Wv)   # new-token V projection (appended to cache)
    Kc = rng.standard_normal((T, d_model))     # T cached keys (already in memory)
    Vc = rng.standard_normal((T, d_model))     # T cached values (already in memory)
    ctx1 = np.empty((1, d_model))
    for h in range(n_heads):
        sl = slice(h * dh, (h + 1) * dh)
        sc = mm(q[:, sl], Kc[:, sl].T)         # (1, T)
        ctx1[:, sl] = mm(sc, Vc[:, sl])        # (1, dh)
    o = mm(ctx1, Wo)
    hup = mm(o, Wup)
    mm(hup, Wdn)
    decode = flops[0]

    return prefill, decode


def grade(sol, fx) -> dict:
    # (d_model, n_heads, d_ff, P, T) — n_heads must divide d_model
    cases = [
        (64, 8, 256, 16, 32),
        (128, 8, 512, 64, 128),
        (256, 16, 1024, 128, 512),
        (64, 1, 128, 1, 1),
        (512, 8, 2048, 256, 1024),
    ]

    exact = 1.0
    max_ratio_err = 0.0
    for d_model, n_heads, d_ff, P, T in cases:
        ref_prefill, ref_decode = _oracle_flops(d_model, n_heads, d_ff, P, T)
        ref_ratio = ref_prefill / ref_decode
        try:
            got = sol.prefill_vs_decode_flops(d_model, n_heads, d_ff, P, T)
            got_prefill = got["prefill"]
            got_decode = got["decode"]
            got_ratio = float(got["ratio"])
        except Exception:
            return {"exact_match": 0.0, "ratio_rel_err": 1.0}

        counts_ok = (
            isinstance(got_prefill, (int, np.integer))
            and isinstance(got_decode, (int, np.integer))
            and int(got_prefill) == ref_prefill
            and int(got_decode) == ref_decode
        )
        if not counts_ok:
            exact = 0.0

        err = abs(got_ratio - ref_ratio) / abs(ref_ratio)
        if err > max_ratio_err:
            max_ratio_err = err

    return {"exact_match": exact, "ratio_rel_err": float(max_ratio_err)}
