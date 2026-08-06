import ref


def check(workdir):
    from exportgate.fidelity import check_fidelity

    out = {"fidelity_matched": 0.0, "tests": float(len(ref.FIDELITY_TESTS))}
    ok = 0
    for i, t in enumerate(ref.FIDELITY_TESTS):
        try:
            res = check_fidelity(t["eager"], t["ort"], rtol=t["rtol"], atol=t["atol"])
            if bool(res) == t["expected"]:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"test {i}: got {res}, expected {t['expected']}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"test {i} raised {type(e).__name__}: {str(e)[:100]}"
    out["fidelity_matched"] = float(ok)
    return out
