CASES = [
    (8, 2, 30),
    (16, 0, 50),
    (16, 4, 60),
    (32, 6, 90),
    (64, 8, 300),
]

CONFIGS = [
    {"n_layers": 12, "n_heads": 8, "n_kv_heads": 8, "head_dim": 64, "n_ctx": 4096, "bytes_per_element": 2},
    {"n_layers": 12, "n_heads": 8, "n_kv_heads": 2, "head_dim": 64, "n_ctx": 4096, "bytes_per_element": 2},
    {"n_layers": 32, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128, "n_ctx": 8192, "bytes_per_element": 2},
    {"n_layers": 24, "n_heads": 16, "n_kv_heads": 1, "head_dim": 128, "n_ctx": 2048, "bytes_per_element": 4},
]


def discard_count(n_past, n_keep):
    n_left = n_past - n_keep
    if n_left < 0:
        n_left = 0
    return n_left // 2


def simulate(n_ctx, n_keep, n_tokens):
    resident = []
    evicted_all = []
    shift_events = []
    next_id = 0
    for _ in range(n_tokens):
        evicted = []
        if len(resident) + 1 > n_ctx:
            n_discard = discard_count(len(resident), n_keep)
            evicted = resident[n_keep:n_keep + n_discard]
            resident = resident[:n_keep] + resident[n_keep + n_discard:]
        resident.append(next_id)
        next_id += 1
        evicted_all.extend(evicted)
        shift_events.append(len(evicted))
    return {"resident": resident, "evicted": evicted_all, "shift_events": shift_events}


def kv_cache_bytes(config):
    return (config["n_layers"] * config["n_ctx"] * 2 *
            config["n_kv_heads"] * config["head_dim"] * config["bytes_per_element"])


def mha_vs_gqa(config):
    mha_config = dict(config)
    mha_config["n_kv_heads"] = config["n_heads"]
    mha_bytes = kv_cache_bytes(mha_config)
    gqa_bytes = kv_cache_bytes(config)
    return {"mha_bytes": mha_bytes, "gqa_bytes": gqa_bytes, "saved_bytes": mha_bytes - gqa_bytes}
