import ref

def check(workdir):
    from bnbdiag.parser import parse_diagnostic
    out = {"dumps_matched": 0.0, "total": float(len(ref.DUMPS))}
    ok = 0
    for i, (dump_text, expected) in enumerate(ref.DUMPS):
        got = parse_diagnostic(dump_text)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"dump {i}: got {got}, expected {expected}"
    out["dumps_matched"] = float(ok)
    return out
