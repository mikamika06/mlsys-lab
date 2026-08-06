CONFIGS = [
    {"num_tokens": 1024, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "f16"},
    {"num_tokens": 1024, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "q8_0"},
    {"num_tokens": 1024, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype": "q4_0"},
    {"num_tokens": 4096, "num_layers": 16, "num_kv_heads": 4, "head_dim": 64, "dtype": "f16"},
    {"num_tokens": 4096, "num_layers": 16, "num_kv_heads": 4, "head_dim": 64, "dtype": "q8_0"},
    {"num_tokens": 4096, "num_layers": 16, "num_kv_heads": 4, "head_dim": 64, "dtype": "q4_0"},
    {"num_tokens": 8192, "num_layers": 28, "num_kv_heads": 4, "head_dim": 128, "dtype": "q4_0"},
]
