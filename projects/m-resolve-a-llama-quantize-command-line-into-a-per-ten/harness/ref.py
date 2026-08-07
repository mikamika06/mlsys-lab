MODELS = [
    {
        "tensors": {
            "blk.0.attn_q.weight": [64, 64],
            "blk.0.attn_v.weight": [64, 64],
            "output.weight": [64, 64],
        },
        "default": "Q4_0",
        "overrides": {"attn_v": "Q8_0"},
        "args": ["--tensor-type", "attn_v=Q8_0"],
    },
    {
        "tensors": {
            "token_embd.weight": [128, 32],
            "blk.0.ffn_gate.weight": [128, 64],
            "blk.0.attn_v.weight": [32, 32],
        },
        "default": "Q8_0",
        "overrides": {"attn_v": "F16", "ffn_gate": "Q4_0"},
        "args": ["--tensor-type", "attn_v=F16", "--tensor-type", "ffn_gate=Q4_0"],
    },
    {
        "tensors": {
            "output_norm.weight": [32],
            "blk.1.attn_v.weight": [32, 32],
            "blk.1.attn_k.weight": [32, 32],
        },
        "default": "F16",
        "overrides": {"attn_v": "Q8_0"},
        "args": ["--tensor-type", "attn_v=Q8_0"],
    },
]
