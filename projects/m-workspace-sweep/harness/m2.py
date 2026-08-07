import sys
import os
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        from sweep.engine import sweep_workspace
    except ImportError as e:
        return {"_note": f"Import failed: {e}"}

    ok = 0
    out = {}

    for i, (config, profile, dev_mem, limits) in enumerate(ref.SCENARIOS):
        want = ref.sweep_workspace(config, profile, dev_mem, limits)
        try:
            got = sweep_workspace(config, profile, dev_mem, limits)
        except Exception as e:
            return {"_note": f"Exception on scenario {i}: {e}"}

        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got index {got}, want {want}"

    out["matches"] = float(ok)
    out["total"] = float(len(ref.SCENARIOS))
    return out
