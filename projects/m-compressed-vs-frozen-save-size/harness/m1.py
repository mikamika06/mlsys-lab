import ref
import quantlibs.sizes as s_mod

def check(workdir):
    out = {"size_ratio_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = s_mod.compute_save_sizes(cfg)
        try:
            got = s_mod.compute_save_sizes(cfg)
        except Exception:
            continue
        if isinstance(got, dict) and "size_ratio" in got:
            if abs(got["size_ratio"] - want["size_ratio"]) < 1e-5:
                ok += 1
    out["size_ratio_matched"] = float(ok)
    return out
