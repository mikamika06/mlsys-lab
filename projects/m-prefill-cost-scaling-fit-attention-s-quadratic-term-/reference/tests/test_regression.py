import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prefill.memory import cheapest_config

def test_kv_cache_is_accounted_for():
    model = {"weights_gb": 10.0, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
    gpus = [{"name": "A", "mem_gb": 12, "cost_per_hr": 1.0, "count": 2}]
    res = cheapest_config(model, gpus, 100000, 5.0)
    assert res == ("A", 2), f"Expected ('A', 2), got {res}"
