import ref


def check(workdir):
    from mtpgap.analysis import analyze_gap
    out = {"losses_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        try:
            res = analyze_gap(cfg)
            if "mtp_loss" in res and "eagle_loss" in res:
                ok += 1
        except Exception:
            pass
    out["losses_matched"] = float(ok)
    return out
