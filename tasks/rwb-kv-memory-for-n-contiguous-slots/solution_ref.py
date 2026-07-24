def kv_memory_bytes(cfg: dict, n_slots: int) -> int:
    """
    Correct implementation of the KV memory size calculation.
    All arithmetic is performed with Python integers to avoid overflow issues.
    """
    layers = cfg["layers"]
    kv_heads = cfg["kv_heads"]
    head_dim = cfg["head_dim"]
    dtype_bytes = cfg["dtype_bytes"]
    n_ctx = cfg["n_ctx"]

    slot_bytes = (
        2
        * layers
        * kv_heads
        * head_dim
        * n_ctx
        * dtype_bytes
    )
    total_bytes = n_slots * slot_bytes
    return int(total_bytes)
