import ref
import numpy as np


def check(workdir):
    from quantlib.layout import fix_qzeros

    out = {"zzeros_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = cfg["expected"]
        try:
            got = fix_qzeros(cfg["qzeros"], cfg["group_size"])
            if np.array_equal(got, want):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i} mismatch"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"exception: {e}"
    out["zzeros_matched"] = float(ok)
    return out
