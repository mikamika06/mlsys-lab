import numpy as np
from gguf_be.writer import write_gguf_be as ref_write_gguf_be
from gguf_be.reader import read_gguf_be as ref_read_gguf_be
from gguf_be.zero_copy import extract_tensor_zero_copy as ref_extract_tensor_zero_copy

TEST_CASES_M1 = [
    {
        "model.name": "llama-be-7b",
        "model.layers": 32,
        "general.alignment": 32,
        "quantized": False,
        "rope.freq_base": 10000.0,
        "token.special_ids": [0, 1, 2],
    },
    {
        "general.architecture": "mamba",
        "general.alignment": 64,
        "mamba.d_state": 16,
        "mamba.expand": 2,
        "mamba.vocab_size": 32000,
        "tags": ["big-endian", "test", "v3"],
    },
]


def generate_test_tensors():
    rng = np.random.RandomState(42)
    t1 = rng.randn(4, 8).astype(np.float32)
    t2 = rng.randint(-1000, 1000, size=(16,)).astype(np.int32)
    t3 = rng.randn(2, 4, 4).astype(np.float16)
    t4 = rng.randint(-128, 127, size=(32,)).astype(np.int8)
    return [
        {"name": "blk.0.attn_q.weight", "data": t1, "dtype": "float32"},
        {"name": "blk.0.attn_q.bias", "data": t2, "dtype": "int32"},
        {"name": "blk.0.attn_k.weight", "data": t3, "dtype": "float16"},
        {"name": "blk.0.attn_k.bias", "data": t4, "dtype": "int8"},
    ]


def compare_vals(got, want):
    if isinstance(want, float):
        return isinstance(got, (float, int)) and abs(got - want) < 1e-4
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return False
        return all(compare_vals(g, w) for g, w in zip(got, want))
    return got == want


def compare_arrays(a, b):
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return bool(np.allclose(a, b, atol=1e-3))
    return bool(np.array_equal(a, b))


def compute_byte_exact_fraction(got_bytes, want_bytes):
    if len(got_bytes) != len(want_bytes):
        min_len = min(len(got_bytes), len(want_bytes))
        max_len = max(len(got_bytes), len(want_bytes))
        matches = sum(
            1 for a, b in zip(got_bytes[:min_len], want_bytes[:min_len]) if a == b
        )
        return float(matches) / float(max_len)
    matches = sum(1 for a, b in zip(got_bytes, want_bytes) if a == b)
    return float(matches) / float(len(want_bytes)) if want_bytes else 1.0
