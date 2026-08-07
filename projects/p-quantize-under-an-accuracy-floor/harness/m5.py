import ref
import sys


def check(workdir):
    m = {"target_reached": 0.0}
    sys_path_orig = list(sys.path)
    try:
        sys.path.insert(0, workdir)
        import quant.target as q_target

        ok1 = q_target.check_target(100, 48, 0.95, 0.945, 0.01)
        ok2 = q_target.check_target(100, 70, 0.95, 0.945, 0.01)
        if ok1 is True and ok2 is False:
            m["target_reached"] = 1.0
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path_orig
    return m
