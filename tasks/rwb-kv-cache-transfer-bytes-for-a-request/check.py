def grade(sol, fx) -> dict:
    """
    Grade the student's kv_cache_transfer_bytes function.
    The grader uses a pure Python oracle that recomputes the closed‑form expression.
    """
    cases = [
        (1, 1, 1, 4, 1),          # minimal values
        (12, 16, 64, 4, 128),     # example from task.md
        (24, 32, 256, 8, 512),
        (6, 8, 96, 2, 64),
        (10, 20, 128, 4, 256)
    ]
    ok = 1.0
    for num_layers, num_kv_heads, head_dim, dtype_bytes, seq_len in cases:
        try:
            got = sol.kv_cache_transfer_bytes(num_layers, num_kv_heads,
                                              head_dim, dtype_bytes, seq_len)
        except Exception:
            ok = 0.0
            break
        expected = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
