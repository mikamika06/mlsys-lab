def predict_max_seq_batch(config):
    available = config["budget_bytes"] - config["fixed_bytes"]
    per_token_batch = (
        2
        * config["layers"]
        * config["kv_heads"]
        * config["head_dim"]
        * config["bytes_per_element"]
    )

    result = []
    for batch in range(1, config["max_batch"] + 1):
        best = 0
        for seq in range(1, config["max_seq"] + 1):
            if batch * seq * per_token_batch <= available:
                best = seq
            else:
                break
        result.append({"batch": batch, "max_seq": best})
    return result
