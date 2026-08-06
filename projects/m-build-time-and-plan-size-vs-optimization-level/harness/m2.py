import sys
import os


def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import ref
    sys.path.pop(0)

    sys.path.insert(0, workdir)
    out = {"stages_matched": 0.0}
    try:
        from trtpipe.diagnostics import classify_failure

        correct = 0
        total = len(ref.FAILURE_LOGS)
        for log_str, expected in ref.FAILURE_LOGS:
            got = classify_failure(log_str)
            if got == expected:
                correct += 1
            elif "_note" not in out:
                out["_note"] = f"For log '{log_str}', expected '{expected}', got '{got}'"
        
        if correct == total:
            out["stages_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Error during execution: {type(e).__name__}: {str(e)}"
    finally:
        if sys.path and sys.path[0] == workdir:
            sys.path.pop(0)
    return out
