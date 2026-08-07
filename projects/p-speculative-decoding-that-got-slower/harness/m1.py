import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        import specdec.analyzer as an
    except ImportError:
        return {"acc_ok": 0.0}

    m = {"acc_ok": 0.0}
    trace = ref.generate_trace()
    try:
        val = an.measure_acceptance(trace)
        if abs(val - 0.4375) < 1e-5:
            m["acc_ok"] = 1.0
    except Exception:
        pass
    return m
