import sys
sys.path.insert(0, ".")
from bakeoff.engine import BakeoffEngine


def test_engine_initialization():
    engine = BakeoffEngine({"seed": 42})
    assert engine.backends is not None
    w = engine.get_weights("baseline")
    assert w is not None


def test_benchmark_outputs():
    engine = BakeoffEngine({"seed": 42})
    res = engine.run_benchmark(runs=2)
    assert len(res) == 3
    for b in res:
        assert "mean_time" in res[b]
