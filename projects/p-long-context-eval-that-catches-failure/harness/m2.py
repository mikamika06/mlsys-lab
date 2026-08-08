import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from longctx.generator import generate_tasks
    from longctx.evaluator import evaluate_curve, detect_dip

    m = {"curve_computed": 0.0, "dip_detected": 0.0}
    try:
        tasks = generate_tasks(1000, 10)
        curve = evaluate_curve(tasks, model_type="flawed")
        if isinstance(curve, list) and len(curve) == 10:
            m["curve_computed"] = 1.0
        if detect_dip(curve):
            m["dip_detected"] = 1.0
    except Exception:
        pass
    return m
