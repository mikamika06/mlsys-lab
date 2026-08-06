import sys

sys.path.insert(0, ".")
from quant.gate import find_smallest_file
from quant.override import override_kv_config
from quant.pipeline import run_pipeline


def test_find_smallest_file_filters_and_minimizes():
    candidates = [
        {"name": "q4_0", "size": 100, "kld": 0.01},
        {"name": "q5_0", "size": 120, "kld": 0.005},
        {"name": "q8_0", "size": 200, "kld": 0.001},
    ]
    best = find_smallest_file(candidates, 0.012)
    assert best["name"] == "q4_0"


def test_override_kv_config_applies_correctly():
    cfg = {"kv_config": {"head_dim": 64, "kv_heads": 4}}
    overrides = {"head_dim": 128}
    res = override_kv_config(cfg, overrides)
    assert res["kv_config"]["head_dim"] == 128
    assert res["kv_config"]["kv_heads"] == 4


def test_run_pipeline_integrates_both():
    candidates = [
        {"name": "q4_k", "size": 90, "kld": 0.02, "config": {"kv_config": {"head_dim": 64}}},
        {"name": "q5_k", "size": 110, "kld": 0.008, "config": {"kv_config": {"head_dim": 64}}},
    ]
    res = run_pipeline(candidates, 0.015, {"head_dim": 128})
    assert res["name"] == "q5_k"
    assert res["config"]["kv_config"]["head_dim"] == 128
