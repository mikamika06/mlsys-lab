CANDIDATES_SET = [
    [
        {"name": "q2_k", "size": 50, "kld": 0.05, "config": {"kv_config": {"head_dim": 64}}},
        {"name": "q4_0", "size": 100, "kld": 0.02, "config": {"kv_config": {"head_dim": 64}}},
        {"name": "q8_0", "size": 200, "kld": 0.001, "config": {"kv_config": {"head_dim": 64}}},
    ],
    [
        {"name": "q3_k", "size": 70, "kld": 0.03, "config": {"kv_config": {"head_dim": 128}}},
        {"name": "q4_k", "size": 95, "kld": 0.015, "config": {"kv_config": {"head_dim": 128}}},
        {"name": "q5_k", "size": 120, "kld": 0.008, "config": {"kv_config": {"head_dim": 128}}},
    ],
    [
        {"name": "q4_0", "size": 110, "kld": 0.004, "config": {"kv_config": {"head_dim": 256}}},
        {"name": "q8_0", "size": 220, "kld": 0.0005, "config": {"kv_config": {"head_dim": 256}}},
    ]
]

OVERRIDES_SET = [
    {"head_dim": 128, "kv_heads": 8},
    {"head_dim": 256, "kv_heads": 4},
    {"head_dim": 64, "kv_heads": 16}
]

MAX_KLDS = [0.025, 0.02, 0.005]
