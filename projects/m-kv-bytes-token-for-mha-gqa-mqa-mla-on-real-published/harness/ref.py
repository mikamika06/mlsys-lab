CONFIGS = [
    {"num_layers": 32, "attn_type": "mha", "num_heads": 32, "head_dim": 128},
    {"num_layers": 32, "attn_type": "gqa", "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    {"num_layers": 32, "attn_type": "mqa", "num_heads": 32, "num_kv_heads": 1, "head_dim": 128},
    {"num_layers": 61, "attn_type": "mla", "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_heads": 128}
]

def calc_bytes_per_token(config, dtype_bytes=2):
    from kvbytes.calc import calc_bytes_per_token as ref_fn
    return ref_fn(config, dtype_bytes)

def measure_growth(config, num_tokens, dtype_bytes=2):
    from kvbytes.measure import measure_growth as ref_fn
    return ref_fn(config, num_tokens, dtype_bytes)

def mla_bytes_per_token(config, dtype_bytes=2):
    from kvbytes.mla import mla_bytes_per_token as ref_fn
    return ref_fn(config, dtype_bytes)
