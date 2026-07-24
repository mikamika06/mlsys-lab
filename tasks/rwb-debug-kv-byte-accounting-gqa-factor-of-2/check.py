import numpy as np


def _ref_bytes(config, seq_len, batch_size) -> int:
    return int(
        2 * batch_size * seq_len * config["num_kv_heads"] * config["head_dim"]
        * config["num_layers"] * config["bytes_per_elem"]
    )


def _scenarios():
    scenarios = []

    # MHA: num_kv_heads == num_attention_heads
    scenarios.append((
        dict(num_attention_heads=32, num_kv_heads=32, head_dim=128, num_layers=32, bytes_per_elem=2),
        4096, 1,
    ))
    # MQA: num_kv_heads = 1
    scenarios.append((
        dict(num_attention_heads=32, num_kv_heads=1, head_dim=128, num_layers=32, bytes_per_elem=2),
        4096, 1,
    ))
    # GQA: group size 4
    scenarios.append((
        dict(num_attention_heads=32, num_kv_heads=8, head_dim=128, num_layers=32, bytes_per_elem=2),
        2048, 4,
    ))
    # GQA: group size 8, int8 cache
    scenarios.append((
        dict(num_attention_heads=64, num_kv_heads=8, head_dim=64, num_layers=48, bytes_per_elem=1),
        8192, 1,
    ))
    # small hand values, batch > 1
    scenarios.append((
        dict(num_attention_heads=16, num_kv_heads=4, head_dim=64, num_layers=12, bytes_per_elem=2),
        512, 8,
    ))

    rng = np.random.default_rng(0)
    for _ in range(6):
        num_kv = int(rng.integers(1, 16))
        group = int(rng.integers(1, 9))
        num_attn = num_kv * group
        head_dim = int(rng.choice([64, 96, 128]))
        num_layers = int(rng.integers(1, 60))
        bytes_per_elem = int(rng.choice([1, 2, 4]))
        seq_len = int(rng.integers(1, 8192))
        batch_size = int(rng.integers(1, 16))
        scenarios.append((
            dict(num_attention_heads=num_attn, num_kv_heads=num_kv, head_dim=head_dim,
                 num_layers=num_layers, bytes_per_elem=bytes_per_elem),
            seq_len, batch_size,
        ))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for config, seq_len, batch_size in _scenarios():
        total += 1
        ref = _ref_bytes(config, seq_len, batch_size)
        try:
            got = sol.kv_cache_bytes(dict(config), seq_len, batch_size)
        except Exception:
            continue
        try:
            if int(got) == ref:
                correct += 1
        except Exception:
            continue

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
