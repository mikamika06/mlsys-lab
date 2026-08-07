import sys
sys.path.insert(0, ".")
from tlog.metrics import compute_percentiles
from tlog.memory import calculate_kv_memory
from tlog.throughput import decode_throughput_ratio


def test_percentiles_basic():
    logs = [
        {"arrival": 0.0, "tokens": [1.0, 1.5, 2.0]}
    ]
    res = compute_percentiles(logs)
    assert res["ttft_p50"] == 1.0
    assert res["itl_p50"] == 0.5


def test_memory_basic():
    res = calculate_kv_memory(num_layers=2, num_kv_heads=4, head_dim=64, max_seq_len=128, dtype_bytes=2, total_memory=1000000)
    assert res["bytes_per_seq"] == 2 * 4 * 64 * 2 * 2 * 128
    assert res["max_sequences"] >= 0


def test_throughput_ratio():
    ratio = decode_throughput_ratio(100.0, 4000.0)
    assert ratio == 40.0
