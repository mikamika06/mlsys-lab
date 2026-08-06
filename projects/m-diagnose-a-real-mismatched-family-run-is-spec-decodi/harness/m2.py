import ref

def check(workdir):
    from specdiag.diagnose import diagnose_run
    out = {"diagnosis_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.diagnose_run(cfg)
        got = diagnose_run(cfg)
        if isinstance(got, dict) and got.get("net_helping") == want["net_helping"] and abs(got.get("speedup", -1) - want["speedup"]) < 1e-5:
            ok += 1
    out["diagnosis_matched"] = float(ok)
    return out
