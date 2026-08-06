import ref


def check(workdir):
    from triton_calc.regs import effective_regs

    out = {"regs_matched": 0.0}
    ok = 0
    for tc in ref.TEST_CASES:
        gran = tc["spec"]["reg_granularity"]
        want = ref.compute_effective_regs(tc["regs_per_thread"], gran)
        try:
            got = effective_regs(tc["regs_per_thread"], gran)
            if got == want:
                ok += 1
        except Exception:
            pass
    out["regs_matched"] = float(ok)
    return out
