import math

def ragged_batched_prefill_attention(Q: list[list[list[float]]], K: list[list[list[float]]], V: list[list[list[float]]], cu_seqlens: list[int]) -> list[list[list[float]]]:
    """
    Q, K, V: (n_tok, n_heads, d) packed tokens from multiple sequences,
        concatenated along axis 0.
    cu_seqlens: 1-D int array of length num_segments + 1; segment s spans
        rows [cu_seqlens[s], cu_seqlens[s + 1]).

    For each segment independently, run causal scaled dot-product
    attention (scale 1/sqrt(d)) restricted to that segment's own rows --
    no query may attend to a key from a different segment.

    Returns a (n_tok, n_heads, d) float64 array.
    """
    raise NotImplementedError('your code here')
