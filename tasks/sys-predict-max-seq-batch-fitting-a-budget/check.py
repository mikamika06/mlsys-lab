def _oracle(config):
    result = []
    available = config["budget_bytes"] - config["fixed_bytes"]
    bytes_per_token = (
        2
        * config["layers"]
        * config["kv_heads"]
        * config["head_dim"]
        * config["bytes_per_element"]
    )

    for batch in range(1, config["max_batch"] + 1):
        best = 0
        for seq in range(1, config["max_seq"] + 1):
            cache_bytes = batch * seq * bytes_per_token
            if cache_bytes <= available:
                best = seq
            else:
                break
        result.append({"batch": batch, "max_seq": best})
    return result


def grade(sol, fx) -> dict:
    cases = [
        {
            "budget_bytes": 100000,
            "fixed_bytes": 10000,
            "layers": 2,
            "kv_heads": 2,
            "head_dim": 4,
            "bytes_per_element": 2,
            "max_batch": 3,
            "max_seq": 100,
        },
        {
            "budget_bytes": 1024 * 1024,
            "fixed_bytes": 200000,
            "layers": 8,
            "kv_heads": 4,
            "head_dim": 64,
            "bytes_per_element": 2,
            "max_batch": 8,
            "max_seq": 512,
        },
        {
            "budget_bytes": 500000,
            "fixed_bytes": 490000,
            "layers": 1,
            "kv_heads": 1,
            "head_dim": 8,
            "bytes_per_element": 4,
            "max_batch": 4,
            "max_seq": 32,
        },
        {
            "budget_bytes": 64 * 1024 * 1024,
            "fixed_bytes": 8 * 1024 * 1024,
            "layers": 12,
            "kv_heads": 8,
            "head_dim": 128,
            "bytes_per_element": 2,
            "max_batch": 16,
            "max_seq": 2048,
        },
    ]

    ok = 1.0
    for case in cases:
        try:
            got = sol.predict_max_seq_batch(dict(case))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(case):
            ok = 0.0
            break
    return {"exact_match": ok}
