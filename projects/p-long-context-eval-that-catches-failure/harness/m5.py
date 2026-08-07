from long_ctx.generator import generate_context_with_fact
from long_ctx.evaluator import evaluate_position_curve

def check(workdir):
    m = {"failure_caught": 0.0}
    try:
        contexts = {0.5: generate_context_with_fact(300, 0.5, "SECRET_FACT")}
        failing_model = lambda text: "Lost"
        curve = evaluate_position_curve(failing_model, contexts)
        if curve.get(0.5) == 0.0:
            m["failure_caught"] = 1.0
    except Exception:
        pass
    return m
