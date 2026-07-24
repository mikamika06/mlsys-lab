def h2d_transfer_bytes(L, layer_bytes, T, max_len):
    """Return (dynamic_bytes, static_bytes) using closed-form formulas."""
    if L + T > max_len:
        raise ValueError("L + T exceeds max_len")
    dynamic = layer_bytes * (T * L + T * (T - 1) // 2)
    static = layer_bytes * (L + T)
    return (dynamic, static)
