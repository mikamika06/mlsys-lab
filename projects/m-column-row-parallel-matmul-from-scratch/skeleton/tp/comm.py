def estimate_tp_layer_comm_bytes(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    ffn_dim: int,
    tp_size: int,
    element_size_bytes: int = 2,
) -> dict:
    raise NotImplementedError
