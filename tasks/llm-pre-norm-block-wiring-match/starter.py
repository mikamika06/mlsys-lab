import numpy as np


def pre_norm_block(x, gamma1, beta1, gamma2, beta2,
                   Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """One pre-norm transformer block over a residual stream x of shape (T, d).

    Wire two sublayers on the residual stream:
        h = x + attn(LN1(x))          # attention sublayer, pre-norm + residual
        y = h + mlp(LN2(h))           # MLP sublayer, pre-norm + residual

    Use LayerNorm over the last axis (population variance, eps=1e-5),
    single-head scaled dot-product self-attention (scores / sqrt(d), softmax,
    then @ Wo), and a two-layer MLP with tanh-approximate GELU. Return y of
    shape (T, d).
    """
    raise NotImplementedError("your code here")
