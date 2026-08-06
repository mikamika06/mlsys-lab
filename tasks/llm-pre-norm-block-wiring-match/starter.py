import math

def pre_norm_block(
    x: list[list[float]],
    gamma1: list[float],
    beta1: list[float],
    gamma2: list[float],
    beta2: list[float],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
    Wo: list[list[float]],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
) -> list[list[float]]:
    """One pre-norm transformer block over a residual stream x of shape (T, d).

    Wire two sublayers on the residual stream:
        h = x + attn(LN1(x))          # attention sublayer, pre-norm + residual
        y = h + mlp(LN2(h))           # MLP sublayer, pre-norm + residual

    Use LayerNorm over the last axis (population variance, eps=1e-5),
    single-head scaled dot-product self-attention (scores / sqrt(d), softmax,
    then @ Wo), and a two-layer MLP with tanh-approximate GELU. Return y of
    shape (T, d).
    """
    raise NotImplementedError('your code here')
