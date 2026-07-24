def _oracle(config, bytes_per_flop):
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

    # Solve:
    # 2*n_layers*n_kv_heads*head_dim*s*dtype_bytes =
    # bytes_per_flop*2*n_layers*n_q_heads*head_dim*s^2
    break_even = (n_kv_heads * dtype_bytes) / (
        n_q_heads * bytes_per_flop
    )

    return int(transfer_bytes), float(break_even)


def grade(sol, fx) -> dict:
    cases = [
        (
            {
                "n_layers": 32,
                "n_kv_heads": 8,
                "n_q_heads": 32,
                "head_dim": 128,
                "seq_len": 4096,
                "dtype_bytes": 2,
            },
            0.25,
        ),
        (
            {
                "n_layers": 80,
                "n_kv_heads": 8,
                "n_q_heads": 64,
                "head_dim": 128,
                "seq_len": 8192,
                "dtype_bytes": 2,
            },
            0.125,
        ),
        (
            {
                "n_layers": 24,
                "n_kv_heads": 24,
                "n_q_heads": 24,
                "head_dim": 64,
                "seq_len": 2048,
                "dtype_bytes": 1,
            },
            0.5,
        ),
    ]

    exact = 1.0
    max_rel = 0.0

    for config, ratio in cases:
        try:
            got_bytes, got_break = sol.kv_transfer_analysis(dict(config), ratio)
        except Exception:
            return {"exact_match": 0.0, "rel_err": 1.0}

        ref_bytes, ref_break = _oracle(config, ratio)

        if int(got_bytes) != ref_bytes:
            exact = 0.0

        rel = abs(float(got_break) - ref_break) / (abs(ref_break) + 1e-12)
        max_rel = max(max_rel, rel)

    return {
        "exact_match": exact,
        "rel_err": 1.0 if max_rel > 1e-6 else 0.0,
    }
