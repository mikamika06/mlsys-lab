import numpy as np


TEST_CASES_MEMORY = [
    {"num_samples": 1000, "seq_len": 512, "vocab_size": 32000, "dtype_bytes": 4},
    {"num_samples": 5000, "seq_len": 1024, "vocab_size": 50257, "dtype_bytes": 2},
    {"num_samples": 10000, "seq_len": 2048, "vocab_size": 128000, "dtype_bytes": 1},
]

TEST_CASES_TOPK = [
    {"num_samples": 1000, "seq_len": 512, "top_k": 8, "logit_bytes": 2, "idx_bytes": 2},
    {"num_samples": 5000, "seq_len": 1024, "top_k": 32, "logit_bytes": 4, "idx_bytes": 4},
]

TEST_CASES_BUDGET = [
    {"ram_budget_bytes": 1073741824, "seq_len": 512, "vocab_size": 32000, "dtype_bytes": 4},
    {"ram_budget_bytes": 8589934592, "seq_len": 1024, "vocab_size": 50257, "dtype_bytes": 2},
]


def ref_compute_full_vocab_footprint(num_samples, seq_len, vocab_size, dtype_bytes=4):
    return int(num_samples * seq_len * vocab_size * dtype_bytes)


def ref_compute_topk_footprint(num_samples, seq_len, top_k, logit_dtype_bytes=2, index_dtype_bytes=2):
    return int(num_samples * seq_len * top_k * (logit_dtype_bytes + index_dtype_bytes))


def ref_max_samples_within_budget(ram_budget_bytes, seq_len, vocab_size, dtype_bytes=4):
    bytes_per_sample = seq_len * vocab_size * dtype_bytes
    if bytes_per_sample <= 0:
        return 0
    return int(ram_budget_bytes // bytes_per_sample)


def make_dummy_dataset(num_samples=10, seq_len=8, seed=42):
    rng = np.random.RandomState(seed)
    dataset = []
    for i in range(num_samples):
        inp = rng.randn(seq_len).astype(np.float32)
        dataset.append({"id": i, "input": inp})
    return dataset
