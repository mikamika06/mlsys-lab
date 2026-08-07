import tempfile
import os


def check(workdir):
    from megacache.artifacts import save_artifact, load_artifact
    out = {"artifacts_matched": 0.0}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "test.pkl")
            art = {"compiled": True, "weights": [1, 2, 3]}
            save_artifact(p, art)
            got = load_artifact(p)
            if got == art:
                out["artifacts_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {e}"
    return out
