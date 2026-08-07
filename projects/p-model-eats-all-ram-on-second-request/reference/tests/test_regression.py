import sys
sys.path.insert(0, ".")
from runner.memory import ModelMemoryProfiler, analyze_duplicates, optimize_config
from runner.config import get_runtime_config

def test_memory_profile_positive():
    prof = ModelMemoryProfiler(4000, 32, 4096, 32)
    assert prof.measure_footprint() > 4000

def test_kv_scaling():
    prof = ModelMemoryProfiler(4000, 32, 4096, 32)
    sz1 = prof.kv_cache_size_mb(1024, 1)
    sz2 = prof.kv_cache_size_mb(2048, 2)
    assert sz2 > sz1

def test_no_duplicates():
    res = analyze_duplicates([{"pid": 101, "cmd": "server"}, {"pid": 101, "cmd": "server"}])
    assert res["duplicate_detected"] is True

def test_config_budget():
    cfg = optimize_config(8192, 4000, 32, 4096, 32)
    assert cfg["slots"] >= 2

def test_runtime_config_valid():
    cfg = get_runtime_config()
    assert cfg["swap_allowed"] is False
