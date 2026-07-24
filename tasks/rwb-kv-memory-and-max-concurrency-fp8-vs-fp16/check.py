import numpy as np


def _ref(config, vram_budget_bytes):
    base = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"]
    bpt_fp8 = base * 1
    bpt_fp16 = base * 2
    seq_len = config["seq_len"]
    max_fp8 = vram_budget_bytes // (bpt_fp8 * seq_len)
    max_fp16 = vram_budget_bytes // (bpt_fp16 * seq_len)
    return {
        "bytes_per_token_fp8": int(bpt_fp8),
        "bytes_per_token_fp16": int(bpt_fp16),
        "max_concurrent_fp8": int(max_fp8),
        "max_concurrent_fp16": int(max_fp16),
    }


def _scenarios():
    scenarios = []
    scenarios.append((
        dict(num_layers=32, num_kv_heads=8, head_dim=128, seq_len=4096),
        40 * 1024**3,
    ))
    scenarios.append((
        dict(num_layers=1, num_kv_heads=1, head_dim=1, seq_len=1),
        1,
    ))
    scenarios.append((
        dict(num_layers=48, num_kv_heads=8, head_dim=64, seq_len=8192),
        24 * 1024**3,
    ))
    scenarios.append((
        dict(num_layers=24, num_kv_heads=32, head_dim=128, seq_len=2048),
        1000,  # tiny budget -> zero concurrency for both
    ))

    rng = np.random.default_rng(0)
    for _ in range(6):
        config = dict(
            num_layers=int(rng.integers(1, 80)),
            num_kv_heads=int(rng.integers(1, 64)),
            head_dim=int(rng.choice([32, 64, 96, 128])),
            seq_len=int(rng.integers(1, 16384)),
        )
        vram_budget_bytes = int(rng.integers(1, 80 * 1024**3))
        scenarios.append((config, vram_budget_bytes))

    return scenarios


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for config, vram_budget_bytes in _scenarios():
        total += 1
        ref = _ref(config, vram_budget_bytes)
        try:
            got = sol.kv_capacity(dict(config), vram_budget_bytes)
        except Exception:
            continue
        try:
            got_norm = {
                "bytes_per_token_fp8": int(got["bytes_per_token_fp8"]),
                "bytes_per_token_fp16": int(got["bytes_per_token_fp16"]),
                "max_concurrent_fp8": int(got["max_concurrent_fp8"]),
                "max_concurrent_fp16": int(got["max_concurrent_fp16"]),
            }
        except Exception:
            continue
        if got_norm == ref:
            correct += 1

    exact_match = (correct / total) if total else 0.0
    return {"exact_match": exact_match}
