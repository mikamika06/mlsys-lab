import ref

def check(workdir):
    from compile_peft.breakeven import break_even_steps
    out = {"breakeven_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS_BREAKEVEN:
        want = ref.calc_break_even(cfg)
        got = break_even_steps(cfg)
        if got == want:
            ok += 1
    out["breakeven_matched"] = 1.0 if ok == len(ref.CONFIGS_BREAKEVEN) else 0.0
    return out
