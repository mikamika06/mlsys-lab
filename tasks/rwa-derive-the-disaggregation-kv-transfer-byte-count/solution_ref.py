def kv_transfer_analysis(config, bytes_per_flop):
    n_layers = int(config["n_layers"])
    n_kv_heads = int(config["n_kv_heads"])
    n_q_heads = int(config["n_q_heads"])
    head_dim = int(config["head_dim"])
    seq_len = int(config["seq_len"])
    dtype_bytes = int(config["dtype_bytes"])

    transfer_bytes = (
        2
        * n_layers
        * n_kv_heads
        * head_dim
        * seq_len
        * dtype_bytes
    )

    break_even_seq_len = (n_kv_heads * dtype_bytes) / (
        n_q_heads * bytes_per_flop
    )

    return int(transfer_bytes), float(break_even_seq_len)
