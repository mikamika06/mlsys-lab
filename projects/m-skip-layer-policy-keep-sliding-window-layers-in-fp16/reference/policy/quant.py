def assign_kv_dtypes(layers):
    return ["float16" if l["kind"] == "sliding" else "float8" for l in layers]

def compute_kv_bytes(layers, dtypes, seq_len, batch_size, kv_heads, head_dim):
    total = 0
    for l, d in zip(layers, dtypes):
        bpe = 4 if d == "float32" else (2 if d == "float16" else 1)
        elements = min(seq_len, l["window"]) if l["kind"] == "sliding" else seq_len
        total += 2 * batch_size * kv_heads * head_dim * elements * bpe
    return total
