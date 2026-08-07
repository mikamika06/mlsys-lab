import ref


def check(workdir):
    from tritonop.parser import parse_kernel

    out = {"parsed_match": 0.0}
    ok = 0
    for code in ref.KERNELS:
        want = ref.parse_kernel(code)
        try:
            got = parse_kernel(code)
        except Exception:
            got = None
        if got == want:
            ok += 1
    out["parsed_match"] = float(ok)
    return out
