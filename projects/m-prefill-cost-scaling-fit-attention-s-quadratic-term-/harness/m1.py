import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from prefill.scaling import fit_scaling
        from prefill.rope import diagnose_rope
    except ImportError:
        return {"scaling_rel_err": 1.0, "rope_matched": 0.0}

    out = {"scaling_rel_err": 1.0, "rope_matched": 0.0}
    errs = []

    for lengths, times in ref.SCALING_FIXTURES:
        want = ref.fit_scaling(lengths, times)
        try:
            got = fit_scaling(lengths, times)
            if not got or "linear" not in got or "quadratic" not in got:
                errs.append(1.0)
                continue
            err_l = abs(want["linear"] - got["linear"]) / max(abs(want["linear"]), 1e-9)
            err_q = abs(want["quadratic"] - got["quadratic"]) / max(abs(want["quadratic"]), 1e-9)
            errs.append(err_l + err_q)
        except Exception:
            errs.append(1.0)

    if errs:
        out["scaling_rel_err"] = float(sum(errs) / len(errs))

    ok_rope = 0
    for cfg, freq in ref.ROPE_FIXTURES:
        want = ref.diagnose_rope(cfg, freq)
        try:
            got = diagnose_rope(cfg, freq)
            if want == got:
                ok_rope += 1
        except Exception:
            pass

    out["rope_matched"] = float(ok_rope)
    return out
