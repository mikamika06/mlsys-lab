import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        import specdec.analyzer as an
    except ImportError:
        return {"speedup_ok": 0.0, "speedup_p1_ok": 0.0}

    m = {"speedup_ok": 0.0, "speedup_p1_ok": 0.0}
    try:
        s1 = an.compute_speedup(0.5, 4, 3.0, 10.0, 11.0)
        if abs(s1 - 0.842391304) < 1e-4:
            m["speedup_ok"] = 1.0

        s2 = an.compute_speedup(1.0, 4, 3.0, 10.0, 11.0)
        if abs(s2 - 2.173913) < 1e-4:
            m["speedup_p1_ok"] = 1.0
    except Exception:
        pass
    return m
