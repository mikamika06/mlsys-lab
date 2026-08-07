import ref


def check(workdir):
    from tritonop.chain import reconstruct_chain

    out = {"chain_match": 0.0}
    ok = 0
    for code in ref.KERNELS:
        want = ref.reconstruct_chain(code)
        try:
            got = reconstruct_chain(code)
        except Exception:
            got = None
        if got == want:
            ok += 1
    out["chain_match"] = float(ok)
    return out
