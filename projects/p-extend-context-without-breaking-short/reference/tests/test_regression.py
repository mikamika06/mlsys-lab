import sys
sys.path.insert(0, ".")
from ctx.scaling import measure_short_degradation, verify_retrieval

def test_short_prompts_match_baseline():
    baseline = 0.95
    current_model = lambda x: 0.96
    score = measure_short_degradation(current_model, [1, 2, 3])
    assert score >= baseline * 0.98, f"Short prompt regression: {score}"

def test_long_prompts_exceed_baseline():
    ctx = "long context data..."
    query = "needle"
    model = lambda c, q: {"found": 1.0}
    assert verify_retrieval(model, ctx, query) > 0.8
