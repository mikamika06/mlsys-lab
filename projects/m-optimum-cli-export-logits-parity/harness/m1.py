import ref
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"parity_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    try:
        from optimum_export.export_utils import verify_logits_parity
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.verify_logits_parity(cfg["hf"], cfg["ov"], cfg["threshold"])
        try:
            got = verify_logits_parity(cfg["hf"], cfg["ov"], cfg["threshold"])
        except Exception as e:
            out["_note"] = f"config {i} raised {e}"
            return out
        if got == want:
            ok += 1
    out["parity_matched"] = float(ok)
    return out
