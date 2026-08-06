import sys

sys.path.insert(0, ".")
from compress.measure import parse_metadata, simulate_load_memory

SAMPLE_CONFIG = {"name": "test_cfg", "bits": 4, "group_size": 128, "numel": 10000, "sparsity": 0.1}


def test_parse_metadata_structure():
    meta = parse_metadata(SAMPLE_CONFIG)
    assert "effective_numel" in meta
    assert "theoretical_bytes" in meta
    assert meta["effective_numel"] > 0


def test_load_memory_non_negative():
    mem = simulate_load_memory(SAMPLE_CONFIG)
    assert mem > 0, "load memory must be positive"


def test_load_memory_scales_with_numel():
    cfg1 = {"name": "c1", "bits": 8, "group_size": 64, "numel": 1000, "sparsity": 0.0}
    cfg2 = {"name": "c2", "bits": 8, "group_size": 64, "numel": 2000, "sparsity": 0.0}
    mem1 = simulate_load_memory(cfg1)
    mem2 = simulate_load_memory(cfg2)
    assert mem2 > mem1, "load memory should scale with number of elements"
