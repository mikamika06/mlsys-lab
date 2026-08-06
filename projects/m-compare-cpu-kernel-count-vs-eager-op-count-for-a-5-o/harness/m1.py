import ref


def check(workdir):
    out = {"ratio_matched": 0.0}
    try:
        from audit.core import analyze_pointwise_chain
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    cases, _ = ref.generate_fixtures()
    ok = 0
    for i, case in enumerate(cases):
        want = ref.analyze_pointwise_chain(case)
        try:
            got = analyze_pointwise_chain(case)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}"
            continue
        if got == want:
            ok += 1
    out["ratio_matched"] = float(ok)
    return out
