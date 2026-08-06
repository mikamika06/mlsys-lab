import sys

sys.path.insert(0, ".")
from runner_map.mapping import map_option
from runner_map.ttl import simulate_ttl
from runner_map.jit import ModelRunner


def test_mapping_supported():
    res = map_option("num_ctx")
    assert res["supported"] is True
    assert res["equivalent"] == "context_length"


def test_mapping_unsupported():
    res = map_option("tfs_z")
    assert res["supported"] is False


def test_ttl_eviction():
    states = simulate_ttl(300, [0, 100, 250, 600])
    assert states == ["loaded", "loaded", "loaded", "evicted"]


def test_jit_loading():
    runner = ModelRunner()
    r1 = runner.request({"prompt": "hello"})
    assert r1["load_count"] == 1
    r2 = runner.request({"prompt": "world"})
    assert r2["load_count"] == 1
