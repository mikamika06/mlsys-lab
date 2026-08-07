import ref
import math
import sys

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from perf import analyzer
    except ImportError:
        return {"metrics_match": 0.0, "_note": "perf.analyzer not found"}

    out = {"metrics_match": 0.0}

    traces = [
        ref.generate_valid_trace(100, 1, 0.001, 0.010, 0.0),
        ref.generate_valid_trace(200, 4, 0.002, 0.020, 0.005),
        ref.generate_inconsistent_trace(),
    ]

    ok = 0
    for i, trace in enumerate(traces):
        r_t, r_hl, r_dl, r_e2e = ref.compute_metrics(trace)
        try:
            got = analyzer.compute_metrics(trace)
        except Exception as e:
            out["_note"] = f"trace {i} raised {e}"
            continue

        if isinstance(got, tuple) and len(got) == 4:
            g_t, g_hl, g_dl, g_e2e = got
            if (math.isclose(r_t, g_t, rel_tol=1e-4, abs_tol=1e-9) and
                math.isclose(r_hl, g_hl, rel_tol=1e-4, abs_tol=1e-9) and
                math.isclose(r_dl, g_dl, rel_tol=1e-4, abs_tol=1e-9) and
                math.isclose(r_e2e, g_e2e, rel_tol=1e-4, abs_tol=1e-9)):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"trace {i}: got {got}, want {(r_t, r_hl, r_dl, r_e2e)}"
        else:
            if "_note" not in out:
                out["_note"] = f"trace {i}: expected tuple length 4, got {got}"

    out["metrics_match"] = float(ok)
    return out
