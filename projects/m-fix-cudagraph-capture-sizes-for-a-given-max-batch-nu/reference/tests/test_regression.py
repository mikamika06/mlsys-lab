import sys

sys.path.insert(0, ".")
from speculative.cudagraph import compute_capture_sizes
from speculative.eagle import find_optimal_eagle_config
from speculative.trtlog import extract_draft_engine_stats


def test_compute_capture_sizes_bounded():
    sizes = compute_capture_sizes(16, 3)
    assert len(sizes) > 0
    assert max(sizes) == 64


def test_find_optimal_eagle_config_valid():
    configs = [
        {"name": "c1", "kv_bytes": 1000, "throughput_score": 10.0},
        {"name": "c2", "kv_bytes": 500, "throughput_score": 8.0},
    ]
    best = find_optimal_eagle_config(configs, 1200)
    assert best["name"] == "c1"


def test_extract_draft_engine_stats_parsing():
    log = "Draft engine avg latency: 12.5 ms\nDraft acceptance rate: 0.85\nDraft engine peak memory: 1048576 bytes"
    st = extract_draft_engine_stats(log)
    assert st["avg_latency_ms"] == 12.5
    assert st["acceptance_rate"] == 0.85
    assert st["peak_memory_bytes"] == 1048576
