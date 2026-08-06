"""Reference data generator and oracle computations."""

import numpy as np


def generate_pairs():
    rng = np.random.default_rng(1337)
    configs = []

    base_vocab = [f"token_{i}" for i in range(100)]
    target_base = {
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tokens": base_vocab,
        "add_eos_token": True,
    }

    c1 = (dict(target_base), dict(target_base), True)
    configs.append(c1)

    d2 = dict(target_base)
    d2["vocab_size"] = 32001
    configs.append((d2, dict(target_base), False))

    d3 = dict(target_base)
    d3["bos_token_id"] = 0
    configs.append((d3, dict(target_base), False))

    d4 = dict(target_base)
    d4["eos_token_id"] = 3
    configs.append((d4, dict(target_base), False))

    d5 = dict(target_base)
    d5["tokens"] = [f"alt_token_{i}" for i in range(100)]
    configs.append((d5, dict(target_base), False))

    d6 = dict(target_base)
    d6["add_eos_token"] = False
    configs.append((d6, dict(target_base), False))

    for _ in range(4):
        vs = int(rng.choice([32000, 64000, 128000]))
        bos = int(rng.integers(1, 5))
        eos = int(rng.integers(5, 10))
        t = dict(target_base)
        t["vocab_size"] = vs
        t["bos_token_id"] = bos
        t["eos_token_id"] = eos
        d = dict(t)
        configs.append((d, t, True))

    return configs


def generate_efficiency_cases():
    return [
        {"r": 0.1, "gamma": 5, "expected_alpha": (5 * 0.1) / (1 + 5 * 0.1)},
        {"r": 0.05, "gamma": 4, "expected_alpha": (4 * 0.05) / (1 + 4 * 0.05)},
        {"r": 0.2, "gamma": 5, "expected_alpha": (5 * 0.2) / (1 + 5 * 0.2)},
        {"r": 0.5, "gamma": 2, "expected_alpha": (2 * 0.5) / (1 + 2 * 0.5)},
        {"r": 0.0, "gamma": 5, "expected_alpha": 0.0},
    ]


def generate_prefill_cases():
    return [
        {"length": 512, "gamma": 5, "draft_t": 0.0002, "target_t": 0.050},
        {"length": 2048, "gamma": 4, "draft_t": 0.00015, "target_t": 0.120},
    ]
