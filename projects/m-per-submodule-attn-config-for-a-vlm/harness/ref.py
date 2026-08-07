CONFIGS = [
    {
        "submodules": [
            {"index": 0, "kind": "vision_proj", "num_heads": 16, "kv_heads": 16, "head_dim": 64, "causal": False},
            {"index": 1, "kind": "vision_proj", "num_heads": 16, "kv_heads": 16, "head_dim": 64, "causal": False},
            {"index": 2, "kind": "text_self", "num_heads": 32, "kv_heads": 8, "head_dim": 128, "causal": True},
            {"index": 3, "kind": "text_self", "num_heads": 32, "kv_heads": 8, "head_dim": 128, "causal": True}
        ]
    },
    {
        "submodules": [
            {"index": 0, "kind": "cross_attn", "num_heads": 16, "kv_heads": 16, "head_dim": 64, "causal": False},
            {"index": 1, "kind": "text_self", "num_heads": 16, "kv_heads": 4, "head_dim": 128, "causal": True},
            {"index": 2, "kind": "text_self", "num_heads": 16, "kv_heads": 4, "head_dim": 128, "causal": True}
        ]
    },
    {
        "submodules": [
            {"index": 0, "kind": "vision_proj", "num_heads": 8, "kv_heads": 8, "head_dim": 64, "causal": False},
            {"index": 1, "kind": "cross_attn", "num_heads": 8, "kv_heads": 8, "head_dim": 64, "causal": False},
            {"index": 2, "kind": "text_self", "num_heads": 16, "kv_heads": 8, "head_dim": 128, "causal": True},
            {"index": 3, "kind": "text_self", "num_heads": 16, "kv_heads": 8, "head_dim": 128, "causal": True},
            {"index": 4, "kind": "pooling", "num_heads": 4, "kv_heads": 4, "head_dim": 128, "causal": False}
        ]
    }
]


def build_configs(config):
    buckets = {}
    for sub in config["submodules"]:
        key = (sub["kind"], sub["num_heads"], sub["kv_heads"], sub["head_dim"], sub["causal"])
        buckets.setdefault(key, []).append(sub["index"])
    out = []
    for (kind, nh, kv, hd, causal), indices in sorted(buckets.items(), key=lambda x: min(x[1])):
        out.append({
            "kind": kind,
            "num_heads": nh,
            "kv_heads": kv,
            "head_dim": hd,
            "causal": causal,
            "submodules": sorted(indices)
        })
    return out


def plan_bytes(config, seq_len, dtype_size, batch_size):
    total = 0
    for sub in config["submodules"]:
        kv_heads = sub["kv_heads"]
        head_dim = sub["head_dim"]
        total += batch_size * seq_len * kv_heads * head_dim * dtype_size * 2
    return total


def uniform_bytes(config, seq_len, dtype_size, batch_size):
    if not config["submodules"]:
        return 0
    max_kv = max(s["kv_heads"] for s in config["submodules"])
    max_hd = max(s["head_dim"] for s in config["submodules"])
    return len(config["submodules"]) * batch_size * seq_len * max_kv * max_hd * dtype_size * 2


def free_schedule(seq_len, dtype_size, step_count):
    res = []
    step_size = max(1, seq_len // step_count)
    current = 0
    for i in range(step_count):
        current = min(seq_len, current + step_size)
        res.append(current * dtype_size * 16)
    return res
