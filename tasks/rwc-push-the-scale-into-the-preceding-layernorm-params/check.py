import numpy as np
from mlsys import scorers

def _ref_layernorm(x, gamma, beta, eps):
    """Standard LayerNorm along the last axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def _oracle_fuse(ln_weight, ln_bias, scale, linear_weight):
    """Oracle: divide LN params by s, multiply each column of W by s."""
    s  = np.asarray(scale,        dtype=np.float64)
    gw = np.asarray(ln_weight,    dtype=np.float64) / s
    gb = np.asarray(ln_bias,      dtype=np.float64) / s
    Wf = np.asarray(linear_weight, dtype=np.float64) * s   # broadcast cols
    return gw, gb, Wf

def grade(sol, fx) -> dict:
    np.random.seed(2024)

    configs = [
        (3, 4, np.array([0.5, 2.0, -1.0])),
        (4, 3, np.array([1.5, 0.3, 2.0, 0.8])),
        (2, 2, np.array([0.1, 10.0])),
        (5, 5, np.array([-0.5, 1.2, 0.01, 5.0, 0.9])),
        (8, 3, np.ones(8)),                       # unit scale — identity check
    ]

    param_errs = []
    output_errs = []

    for d_in, d_out, scale in configs:
        gamma = np.random.randn(d_in) * 2 + 0.5
        beta  = np.random.randn(d_in)
        eps   = 1e-5
        W     = np.random.randn(d_out, d_in)
        b     = np.random.randn(d_out)

        # ---- student call ----
        try:
            sg, sb, sW = sol.fuse_scale_into_layernorm(
                gamma, beta, eps, scale, W, b)
            sg  = np.asarray(sg,  dtype=np.float64)
            sb  = np.asarray(sb,  dtype=np.float64)
            sW  = np.asarray(sW,  dtype=np.float64)
        except Exception:
            return {"ln_param_err": 1.0, "output_err": 1.0}

        # ---- oracle expected ----
        rg, rb, rW = _oracle_fuse(gamma, beta, scale, W)

        # gate 1: LN param accuracy
        pe = max(scorers.rel_err(rg, sg),
                 scorers.rel_err(rb, sb))
        param_errs.append(pe)

        # gate 2: end-to-end output equivalence
        x = np.random.randn(16, d_in)

        # Unfused (original) path:  LN(x, γ, β) → Linear(W, b)
        ln_out      = _ref_layernorm(x, gamma, beta, eps)
        out_unfused = ln_out @ W.T + b

        # Fused (student) path:  LN(x, γ', β') → Linear(W', b)
        ln_fused    = _ref_layernorm(x, sg, sb, eps)
        out_fused   = ln_fused @ sW.T + b

        oe = scorers.rel_err(out_unfused, out_fused)
        output_errs.append(oe)

    return {
        "ln_param_err": max(param_errs),
        "output_err":  max(output_errs),
    }
