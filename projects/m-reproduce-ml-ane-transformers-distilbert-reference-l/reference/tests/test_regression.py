import sys
import numpy as np

sys.path.insert(0, ".")
from transformer.ane_distilbert import evaluate_latencies
from transformer.split_softmax import compute_split_softmax
from transformer.l2_chunk import derive_l2_chunk_size


def test_latency_ordering_keys():
    latencies = evaluate_latencies()
    assert "attention" in latencies
    assert "ffn" in latencies
    assert latencies["attention"] > latencies["layer_norm"]


def test_split_softmax_concatenation():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(1, 4, 16, 32))
    parts = compute_split_softmax(x, chunks=2)
    assert len(parts) == 2
    full_out = np.concatenate(parts, axis=-1)
    assert full_out.shape == x.shape
    assert np.allclose(np.sum(full_out, axis=-1), 1.0, atol=1e-6)


def test_l2_chunk_size_bounds():
    shape = (128, 64)
    elem_size = 2
    l2_cap = 4096
    chunk_size = derive_l2_chunk_size(shape, elem_size, l2_cap)
    assert isinstance(chunk_size, int)
    assert 1 <= chunk_size <= shape[0]

    huge_shape = (64, 8192)
    chunk_huge = derive_l2_chunk_size(huge_shape, elem_size, l2_cap)
    assert 1 <= chunk_huge <= l2_cap // elem_size
