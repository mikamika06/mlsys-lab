import sys

sys.path.insert(0, ".")
from llamaslot.context import compute_slot_context, plan_slot_allocation
from llamaslot.metrics import analyze_prompt_cache, parse_prometheus_metrics
from llamaslot.saturation import find_np_saturation


def test_slot_context_calculation():
    slot_ctx = compute_slot_context(8192, 4, 16384)
    assert slot_ctx == 2048
    plan = plan_slot_allocation(4096, 0, 4, 512)
    assert plan["total_ctx"] == 4096
    assert plan["slot_ctx"] == 1024
    assert plan["is_valid"] is True


def test_zero_ctx_size_defaults_to_model_max():
    slot_ctx = compute_slot_context(0, 2, 4096)
    assert slot_ctx == 2048


def test_np_saturation_bounds():
    sat = find_np_saturation(16384, 2048, 8)
    assert sat["sat_np"] == 8
    assert sat["slot_ctx"] == 2048
    assert sat["slot_ctx"] * sat["sat_np"] <= 16384
    assert sat["is_saturated"] is True


def test_metrics_prompt_cache_hit_ratio():
    raw = """
    llamacpp:prompt_tokens_total 1000.0
    llamacpp:prompt_tokens_processed 200.0
    llamacpp:prompt_tokens_cached 800.0
    """
    res = analyze_prompt_cache(raw)
    assert res["prompt_tokens_total"] == 1000
    assert res["prompt_tokens_cached"] == 800
    assert res["hit_ratio"] == 0.8
    assert res["is_reusing"] is True
