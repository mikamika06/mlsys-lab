import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from longctx.generator import generate_tasks
    from longctx.evaluator import evaluate_curve, detect_dip

    m = {"failure_caught": 0.0}
    try:
        tasks = generate_tasks(2000, 11)
        curve = evaluate_curve(tasks, "flawed")
        if detect_dip(curve):
            m["failure_caught"] = 1.0
    except Exception:
        pass
    return m
