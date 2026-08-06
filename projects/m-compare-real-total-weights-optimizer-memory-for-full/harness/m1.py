import os
import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"params_matched": 0.0}
    try:
        from optmem import params
        configs = ref.get_configs()
        out["configs"] = float(len(configs))
        ok = 0
        for i, cfg in enumerate(configs):
            want = ref.count_parameters(cfg)
            got = params.count_parameters(cfg)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, reference {want}"
        out["params_matched"] = float(ok)
    except Exception as e:
        out["_note"] = f"Error during check: {type(e).__name__}: {e}"
    return out
