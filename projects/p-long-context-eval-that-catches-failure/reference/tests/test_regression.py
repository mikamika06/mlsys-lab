import sys
sys.path.insert(0, ".")
from long_ctx.generator import generate_context_with_fact
from long_ctx.evaluator import evaluate_position_curve
from long_ctx.analyzer import isolate_failure_mode

def test_long_context_retrieval():
    ctx = generate_context_with_fact(1000, 0.5, "SECRET_FACT")
    assert "SECRET_FACT" in ctx

def test_middle_loss_detection():
    mock_model = lambda text: "SECRET_FACT" if "SECRET_FACT" in text else ""
    curve = evaluate_position_curve(mock_model, {0.5: generate_context_with_fact(1000, 0.5, "SECRET_FACT")})
    assert curve[0.5] == 1.0
