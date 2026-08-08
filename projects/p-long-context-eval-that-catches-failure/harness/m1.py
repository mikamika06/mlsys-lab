import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from longctx.generator import generate_tasks

    m = {"tasks_generated": 0.0, "positions_covered": 0.0}
    try:
        tasks = generate_tasks(1000, 5)
        if isinstance(tasks, list) and len(tasks) >= 5:
            m["tasks_generated"] = 1.0
            m["positions_covered"] = float(len(tasks))
    except Exception:
        pass
    return m
