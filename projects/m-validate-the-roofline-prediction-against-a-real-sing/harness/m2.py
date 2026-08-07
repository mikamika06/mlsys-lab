import ref

def check(workdir):
    from roofline.sweep import validate_sweep
    out = {"rel_err_ok": 0.0}
    try:
        sw = ref.SWEEPS[0]
        cfg = ref.CONFIGS[sw["config_idx"]]
        want = ref.validate_sweep(sw, max_rel_err=0.30)
        got = validate_sweep(sw, cfg, max_rel_err=0.30)
        if isinstance(got, dict) and "max_rel_err" in got:
            diff = abs(got["max_rel_err"] - want["max_rel_err"])
            if diff < 1e-3:
                out["rel_err_ok"] = 1.0
            else:
                out["_note"] = f"max_rel_err got {got['max_rel_err']}, reference {want['max_rel_err']}"
        else:
            out["_note"] = "validate_sweep did not return expected dict with max_rel_err"
    except Exception as e:
        out["_note"] = f"validate_sweep raised {type(e).__name__}: {str(e)[:120]}"
    return out
