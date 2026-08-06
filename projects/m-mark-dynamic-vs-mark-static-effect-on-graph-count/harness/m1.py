import ref
import torch


def check(workdir):
    from dynshape.mark import apply_markings

    out = {"markings_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        t = torch.randn(*cfg["shape"])
        try:
            apply_markings(t, cfg["dynamic"], cfg["static"])
            ok += 1
        except Exception:
            pass
    out["markings_matched"] = float(ok)
    return out
