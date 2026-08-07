import ref

def check(workdir):
    from tritonfix.reconstruct import reconstruct_fix
    out = {"exact_matched": 0.0}
    ok = 0
    for logs in ref.LOGS:
        want = ref.reconstruct_fix(logs)
        got = reconstruct_fix(logs)
        if got == want:
            ok += 1
    out["exact_matched"] = float(ok)
    return out
