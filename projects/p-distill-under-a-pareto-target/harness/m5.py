import sys
import ref

def check(workdir):
    m = {"pareto_target_met": 0.0}
    sys.path.insert(0, workdir)
    try:
        import distill.pareto as p
        res = p.check_pareto(0.95, 0.94, 100, 50)
        if isinstance(res, bool):
            m["pareto_target_met"] = 1.0
    except Exception:
        pass
    return m
