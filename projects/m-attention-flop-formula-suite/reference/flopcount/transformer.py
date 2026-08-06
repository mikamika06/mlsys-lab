from flopcount.attention import count_attention_flops


def count_layer_flops(
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ffn_hidden_dim: int,
    causal: bool = True,
    pass_type: str = "fwd",
) -> int:
    """Calculates FLOPs for a single Transformer layer."""
    q_proj = 2 * seq_len * hidden_dim * (num_heads * head_dim)
    k_proj = 2 * seq_len * hidden_dim * (num_kv_heads * head_dim)
    v_proj = 2 * seq_len * hidden_dim * (num_kv_heads * head_dim)
    out_proj = 2 * seq_len * (num_heads * head_dim) * hidden_dim

    attn_core = count_attention_flops(
        b=1,
        h_q=num_heads,
        h_kv=num_kv_heads,
        s_q=seq_len,
        s_k=seq_len,
        d=head_dim,
        causal=causal,
    )

    gate_proj = 2 * seq_len * hidden_dim * ffn_hidden_dim
    up_proj = 2 * seq_len * hidden_dim * ffn_hidden_dim
    down_proj = 2 * seq_len * ffn_hidden_dim * hidden_dim

    fwd_flops = (
        q_proj
        + k_proj
        + v_proj
        + out_proj
        + attn_core
        + gate_proj
        + up_proj
        + down_proj
    )

    if pass_type == "fwd":
        return fwd_flops
    elif pass_type == "bwd":
        return 2 * fwd_flops
    elif pass_type == "fwd_bwd":
        return 3 * fwd_flops
    else:
        raise ValueError(f"Unknown pass_type: {pass_type}")


def count_transformer_flops(
    num_layers: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ffn_hidden_dim: int,
    vocab_size: int,
    causal: bool = True,
    pass_type: str = "fwd",
) -> int:
    """Calculates FLOPs for a full transformer model."""
    layer_flops = count_layer_flops(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        causal=causal,
        pass_type=pass_type,
    )
    total_layers = num_layers * layer_flops

    lm_head_fwd = 2 * seq_len * hidden_dim * vocab_size
    if pass_type == "fwd":
        lm_head_flops = lm_head_fwd
    elif pass_type == "bwd":
        lm_head_flops = 2 * lm_head_fwd
    elif pass_type == "fwd_bwd":
        lm_head_flops = 3 * lm_head_fwd
    else:
        raise ValueError(f"Unknown pass_type: {pass_type}")

    return total_layers + lm_head_flops
