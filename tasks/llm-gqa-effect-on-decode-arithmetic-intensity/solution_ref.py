def decode_arithmetic_intensity(num_heads, head_dim, seq_len, use_gqa):
    """
    Compute the arithmetic intensity for a single decoding step of multi‑head attention.
    
    Parameters
    ----------
    num_heads : int
        Number of query heads in the layer.
    head_dim : int
        Dimensionality of each head (d_k).
    seq_len : int
        Length of the past key/value cache (number of tokens seen so far).
    use_gqa : bool
        If True, grouped‑query attention halves the number of KV heads.
    
    Returns
    -------
    float
        Arithmetic intensity = FLOPs / bytes read.
    """
    kv_heads = num_heads // 2 if use_gqa else num_heads
    ops = num_heads * head_dim * seq_len
    mem_bytes = num_heads * head_dim + 2 * seq_len * kv_heads * head_dim
    return ops / mem_bytes
