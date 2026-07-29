def build_groups(config):
    buckets = {}
    for layer in config["layers"]:
        key = (layer["kind"], layer.get("window") or 0, layer["kv_heads"], layer["head_dim"])
        buckets.setdefault(key, []).append(layer["index"])
    return [{"kind": k[0], "window": k[1], "kv_heads": k[2], "head_dim": k[3],
             "layers": sorted(v)} for k, v in sorted(buckets.items())]


def blocks(tokens, block_size):
    return (tokens + block_size - 1) // block_size


def group_bytes(g, ctx, bs, elem):
    span = ctx if g["kind"] == "full" else min(g["window"], ctx)
    return blocks(span, bs) * bs * 2 * g["kv_heads"] * g["head_dim"] * elem * len(g["layers"])


def plan_bytes(config, ctx, bs, elem):
    return sum(group_bytes(g, ctx, bs, elem) for g in build_groups(config))


def uniform_bytes(config, ctx, bs, elem):
    return sum(blocks(ctx, bs) * bs * 2 * l["kv_heads"] * l["head_dim"] * elem
               for l in config["layers"])


def free_schedule(window, bs, steps):
    return [max(0, t - window) // bs for t in range(1, steps + 1)]


CONFIGS = [
    {"layers": [
        {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 128},
        {"index": 1, "kind": "sliding", "window": 512, "kv_heads": 8, "head_dim": 128},
        {"index": 2, "kind": "sliding", "window": 512, "kv_heads": 8, "head_dim": 128},
        {"index": 3, "kind": "sliding", "window": 2048, "kv_heads": 8, "head_dim": 128},
        {"index": 4, "kind": "full", "kv_heads": 8, "head_dim": 128}]},
    {"layers": [{"index": i, "kind": "sliding" if i % 3 else "full",
                 "window": (256 if i % 2 else 1024) if i % 3 else 0,
                 "kv_heads": 4 if i < 6 else 8, "head_dim": 64}
                for i in range(12)]},
    {"layers": [{"index": i, "kind": "full", "kv_heads": 2, "head_dim": 32} for i in range(4)]},
]
