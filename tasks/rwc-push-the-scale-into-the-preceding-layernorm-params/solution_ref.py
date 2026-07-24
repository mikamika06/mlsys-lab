import numpy as np

def fuse_scale_into_layernorm(ln_weight, ln_bias, ln_eps, scale,
                               linear_weight, linear_bias):
    """Fold per-feature scale into preceding LayerNorm gamma/beta and into
    the next Linear weight.

    Fused identity:  LN(x, γ/s, β/s) → Linear(W·diag(s), b)
    equals:          LN(x, γ,  β)  → Linear(W, b)

    Parameters
    ----------
    ln_weight, ln_bias  : (d_in,) float64  – LayerNorm γ, β
    ln_eps              : float             – LayerNorm ε
    scale               : (d_in,) float64  – per-feature scale s
    linear_weight       : (d_out, d_in)    – Linear W
    linear_bias         : (d_out,)         – Linear b (unchanged, not returned)

    Returns
    -------
    new_ln_weight, new_ln_bias, new_linear_weight
    """
    ln_weight     = np.asarray(ln_weight,     dtype=np.float64)
    ln_bias       = np.asarray(ln_bias,       dtype=np.float64)
    scale         = np.asarray(scale,         dtype=np.float64)
    linear_weight = np.asarray(linear_weight, dtype=np.float64)

    # Fold diag(1/s) into LayerNorm parameters
    new_ln_weight = ln_weight / scale
    new_ln_bias   = ln_bias   / scale

    # Fold diag(s) into Linear weight: column j multiplied by s[j]
    new_linear_weight = linear_weight * scale

    return new_ln_weight, new_ln_bias, new_linear_weight
