import sys
import numpy as np

sys.path.insert(0, ".")
from splitkv.occupancy import compute_split_count, partition_kv_ranges
from splitkv.combine import split_kv_attention
from splitkv.cost import find_optimal_splits


def test_split_and_combine_exactness():
    rng = np.random.RandomState(123)
    q = rng.randn(2, 4, 1, 64)
    k = rng.randn(2, 4, 512, 64)
    v = rng.randn(2, 4, 512, 64)

    d_k = 64
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    m = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - m)
    lse = np.sum(exp_scores, axis=-1, keepdims=True)
    ref_out = np.matmul(exp_scores / lse, v)

    split_out, _ = split_kv_attention(q, k, v, split_count=4)
    err = np.max(np.abs(split_out - ref_out) / (np.abs(ref_out) + 1e-8))
    assert err < 1e-4, f"split attention output err {err} exceeds threshold"


def test_occupancy_split_count_bounds():
    s = compute_split_count(batch_size=1, num_heads=4, kv_len=1024, block_size=128, num_sms=108, target_waves=1, max_splits=16)
    assert 1 <= s <= 8
    ranges = partition_kv_ranges(1024, s)
    assert len(ranges) == s
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 1024


def test_cost_model_optimal_splits():
    opt_s = find_optimal_splits(batch_size=1, num_heads=2, kv_len=8192, block_size=128, head_dim=64, num_sms=108)
    assert opt_s >= 1
