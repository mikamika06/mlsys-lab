import sys
sys.path.insert(0, ".")
from autotune.cache import make_cache_key
from autotune.tuner import Autotuner
from autotune.metrics import measure_latency

def test_cache_keys_differentiate():
    k1 = make_cache_key((128, 128), (128, 1))
    k2 = make_cache_key((64, 256), (256, 1))
    assert k1 != k2

def test_measure_latency_positive():
    val = measure_latency(lambda: sum(range(100)), [], warmup=2, reps=5)
    assert val > 0

def test_tuner_selects_valid():
    configs = [{"block": 32}, {"block": 64}]
    t = Autotuner(configs)
    cfg, lat = t.select([128], [1], lambda c: sum(range(c["block"])))
    assert cfg in configs
    assert lat > 0
