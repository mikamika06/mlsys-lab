TENSORS = [
    ("blk.0.attn_q.weight", 1000),
    ("blk.0.ffn_gate.weight", 2000),
    ("blk.1.attn_q.weight", 1000),
    ("blk.1.ffn_gate.weight", 2000),
]
OVERRIDES = [
    ("blk\\.0\\..*", "CPU"),
    (".*ffn_gate.*", "GPU"),
]
PATTERNS = ["ffn_gate"]
TOTAL_LAYERS = 2
