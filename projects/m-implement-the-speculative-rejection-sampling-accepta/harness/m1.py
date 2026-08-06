import sys
import ref
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from speculative.accept import evaluate_draft
    except ImportError:
        sys.path.pop(0)
        return {"matches": 0.0, "_note": "failed to import evaluate_draft"}

    fixtures = ref.generate_fixtures()
    out = {"matches": 0.0, "total": float(len(fixtures))}
    ok = 0

    for tp, dp, tk, u in fixtures:
        want_n, want_dist = ref.evaluate_draft(tp, dp, tk, u)
        try:
            got_n, got_dist = evaluate_draft(tp, dp, tk, u)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"crash: {e}"
            continue

        if got_n != want_n:
            if "_note" not in out:
                out["_note"] = f"expected {want_n} accepted tokens, got {got_n}"
            continue

        if want_dist is None and got_dist is not None:
            continue
        if want_dist is not None and got_dist is None:
            continue

        if want_dist is not None:
            if not np.allclose(want_dist, got_dist, atol=1e-5):
                continue

        ok += 1

    out["matches"] = float(ok)
    sys.path.pop(0)
    return out
