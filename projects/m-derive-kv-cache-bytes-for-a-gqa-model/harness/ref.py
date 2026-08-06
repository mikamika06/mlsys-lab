CONFIGS = [
    {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
    {"num_layers": 40, "num_kv_heads": 4, "head_dim": 64},
    {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
]

def calc_kv_bytes(config, context_len, dtype_bytes):
    return 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * context_len * dtype_bytes

def calc_quant_kv_bytes(config, context_len, quant_type):
    total_elements = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * context_len
    if quant_type == "F32":
        bpe = 4.0
    elif quant_type == "F16":
        bpe = 2.0
    elif quant_type == "Q8_0":
        bpe = 34.0 / 32.0
    elif quant_type == "Q4_0":
        bpe = 18.0 / 32.0
    else:
        bpe = 2.0
    return int(total_elements * bpe)
