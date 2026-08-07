from kvmodel.sizing import compute_kv_bytes


def dtype_comparison_table(config, seq_len, batch_size):
    dtypes = {"fp16": 2, "fp8": 1, "int4": 0.5}
    table = {}
    for name, b in dtypes.items():
        table[name] = compute_kv_bytes(config, seq_len, batch_size, b)
    return table
