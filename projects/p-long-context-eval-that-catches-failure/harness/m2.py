from long_ctx.generator import generate_context_with_fact
from long_ctx.evaluator import evaluate_position_curve

def check(workdir):
    m = {"curve_computed": 0.0, "dip_detected": 0.0}
    try:
        contexts = {
            0.1: generate_context_with_fact(400, 0.1, "SECRET_FACT"),
            0.5: generate_context_with_fact(400, 0.5, "SECRET_FACT"),
            0.9: generate_context_with_fact(400, 0.9, "SECRET_FACT")
        }
        mock_model = lambda text: "SECRET_FACT" if "SECRET_FACT" in text and "0.5" not in text else "Failed"
        curve = evaluate_position_curve(mock_model, contexts)
        if isinstance(curve, dict) and len(curve) == 3:
            m["curve_computed"] = 1.0
            if curve.get(0.5, 1.0) == 0.0:
                m["dip_detected"] = 1.0
    except Exception:
        pass
    return m
