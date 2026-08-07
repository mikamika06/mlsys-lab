import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from llm_sched.metrics import parse_utilization
        want = ref.parse_utilization(ref.LOG_LINES)
        got = parse_utilization(ref.LOG_LINES)

        mean_err = abs(want["mean"] - got["mean"]) / (want["mean"] + 1e-9)
        max_err = abs(want["max"] - got["max"]) / (want["max"] + 1e-9)

        return {"mean_rel_err": mean_err, "max_rel_err": max_err}
    except Exception as e:
        return {"_note": str(e), "mean_rel_err": 1.0, "max_rel_err": 1.0}
