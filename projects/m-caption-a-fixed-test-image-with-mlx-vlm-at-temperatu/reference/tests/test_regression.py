from mlx_vlm_utils.profile import compare_prompts


def test_compare_prompts_logic():
    s = {"latency": 10.0, "memory": 100.0}
    m = {"latency": 25.0, "memory": 220.0}
    res = compare_prompts(s, m)
    assert res["valid"] is True
    assert res["latency_ratio"] > 1.0
    assert res["memory_ratio"] > 1.0
