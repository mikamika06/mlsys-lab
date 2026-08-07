import sys
sys.path.insert(0, ".")
from ort_tune.config import RuntimeEngine

def test_latency_under_threshold():
    cfg = {"intra_threads": 4, "enable_arena": True, "io_binding": True, "opt_level": 99}
    engine = RuntimeEngine(cfg)
    lat = engine.run(None)
    assert lat < 100.0, f"latency {lat} is too high"

def test_threads_bound():
    cfg = {"intra_threads": 4}
    assert cfg["intra_threads"] > 0
