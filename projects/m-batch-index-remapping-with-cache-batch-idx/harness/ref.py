import numpy as np


def get_mapping_test_cases():
    return [
        {
            "active": ["req0", "req2", "req5"],
            "map": {"req0": 0, "req1": 1, "req2": 4, "req5": 2},
            "max_batch": 8,
            "expected": np.array([0, 4, 2], dtype=np.int32),
        },
        {
            "active": ["reqB", "reqA"],
            "map": {"reqA": 7, "reqB": 3},
            "max_batch": 10,
            "expected": np.array([3, 7], dtype=np.int32),
        },
    ]


def get_invalid_mapping_test_cases():
    return [
        {
            "active": ["req0", "req_missing"],
            "map": {"req0": 0},
            "max_batch": 4,
        },
        {
            "active": ["req0", "req1"],
            "map": {"req0": 5, "req1": 1},
            "max_batch": 4,
        },
        {
            "active": ["req0", "req1"],
            "map": {"req0": 2, "req1": 2},
            "max_batch": 4,
        },
    ]


def get_remap_test_cases():
    return [
        {
            "old_idx": np.array([5, 2, 0, 7], dtype=np.int32),
            "mask": np.array([True, False, True, False]),
            "expected": np.array([5, 0], dtype=np.int32),
        },
        {
            "old_idx": np.array([1, 3], dtype=np.int32),
            "mask": np.array([True, True]),
            "expected": np.array([1, 3], dtype=np.int32),
        },
    ]


def generate_cache_test_scenario(seed=42):
    rng = np.random.RandomState(seed)
    max_cache_batch = 6
    max_seq_len = 16
    num_heads = 4
    head_dim = 8
    batch_size = 3

    cache_batch_idx = np.array([4, 0, 2], dtype=np.int32)
    seq_lens = np.array([3, 5, 1], dtype=np.int32)

    new_k = rng.randn(batch_size, num_heads, head_dim).astype(np.float32)
    new_v = rng.randn(batch_size, num_heads, head_dim).astype(np.float32)

    return {
        "max_cache_batch": max_cache_batch,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "cache_batch_idx": cache_batch_idx,
        "seq_lens": seq_lens,
        "new_k": new_k,
        "new_v": new_v,
    }
