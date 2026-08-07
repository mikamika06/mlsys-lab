def get_layer_dtypes(config, keep_sliding_fp16=True):
    dtypes = {}
    for layer in config["layers"]:
        idx = layer["index"]
        kind = layer.get("kind", "full")
        if keep_sliding_fp16 and kind == "sliding":
            dtypes[idx] = "fp16"
        else:
            dtypes[idx] = "fp8"
    return dtypes


def compute_kv_bytes(config, dtypes, seq_len):
    total_bytes = 0
    for layer in config["layers"]:
        idx = layer["index"]
        kind = layer.get("kind", "full")
        num_heads = layer["num_heads"]
        head_dim = layer["head_dim"]
        dt = dtypes.get(idx, "fp16")
        elem_bytes = 2 if dt == "fp16" else 1
        if kind == "sliding":
            win = layer.get("window", seq_len)
            effective_seq = min(seq_len, win)
        else:
            effective_seq = seq_len
        total_bytes += 2 * effective_seq * num_heads * head_dim * elem_bytes
    return total_bytes
