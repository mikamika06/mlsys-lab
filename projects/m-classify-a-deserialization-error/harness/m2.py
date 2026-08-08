import ref


def check(workdir):
    from plan import diagnose_load

    ok = 0
    out = {}

    for i, fix in enumerate(ref.FIXTURES):
        try:
            got = diagnose_load(fix["engine"], *fix["env"])
            if got.get("status") == fix["want_status"] and abs(got.get("penalty", -1.0) - fix["want_penalty"]) < 1e-5:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"fix {i}: want {fix['want_status']} ({fix['want_penalty']}), got {got.get('status')} ({got.get('penalty')})"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"fix {i} raised {type(e).__name__}"

    out["diagnostics_match"] = float(ok)
    return out
