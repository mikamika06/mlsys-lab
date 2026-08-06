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
    raise NotImplementedError


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
    raise NotImplementedError
