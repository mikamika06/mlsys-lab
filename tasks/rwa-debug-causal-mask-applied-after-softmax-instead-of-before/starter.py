import math

def causal_self_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    """Causal scaled dot-product self-attention.

    Q, K, V: (n, d). Row i may only attend to keys/values at position
    <= i. Returns (n, d).

    BUG: the causal mask is applied to the softmax PROBABILITIES (zeroing
    disallowed entries after the fact) instead of to the logits before
    softmax, so masked rows no longer sum to 1 -- the output is silently
    under-normalized.
    """
    raise NotImplementedError('your code here')
