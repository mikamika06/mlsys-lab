CONFIGS = [
    {
        "id": "llama-3-70b-sim",
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192
    },
    {
        "id": "deepseek-v3-sim",
        "num_hidden_layers": 61,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "hidden_size": 7168
    },
    {
        "id": "custom-gqa",
        "num_layers": 24,
        "num_kv_heads": 4,
        "head_dim": 64
    }
]
