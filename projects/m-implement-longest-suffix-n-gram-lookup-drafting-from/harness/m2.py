import sys
import os
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from prompt_lookup.simulate import simulate
    except ImportError:
        return {"_note": "could not import simulate from prompt_lookup.simulate"}

    out = {"matches": 0.0, "total": float(len(ref.M2_CASES))}
    ok = 0
    for prompt, target, max_n, max_draft_len, want_steps, want_acc in ref.M2_CASES:
        try:
            got = simulate(prompt, target, max_n, max_draft_len)
            if got.get("steps") == want_steps and got.get("accepted") == want_acc:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"got {got.get('steps')} steps and {got.get('accepted')} acc, want {want_steps} and {want_acc}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"crashed: {type(e).__name__}: {str(e)}"

    out["matches"] = float(ok)
    sys.path.pop(0)
    return out
