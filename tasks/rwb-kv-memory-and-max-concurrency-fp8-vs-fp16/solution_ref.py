def kv_capacity(config, vram_budget_bytes):
    base = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"]
    bytes_per_token_fp8 = base * 1
    bytes_per_token_fp16 = base * 2
    seq_len = config["seq_len"]

    max_concurrent_fp8 = vram_budget_bytes // (bytes_per_token_fp8 * seq_len)
    max_concurrent_fp16 = vram_budget_bytes // (bytes_per_token_fp16 * seq_len)

    return {
        "bytes_per_token_fp8": bytes_per_token_fp8,
        "bytes_per_token_fp16": bytes_per_token_fp16,
        "max_concurrent_fp8": max_concurrent_fp8,
        "max_concurrent_fp16": max_concurrent_fp16,
    }
