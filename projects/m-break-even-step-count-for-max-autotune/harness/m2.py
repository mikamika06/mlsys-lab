import ref


def check(workdir):
    from autotune.breakeven import compute_break_even

    cases = ref.generate_breakeven_cases()
    max_err = 0.0
    for overhead, t_def, t_auto in cases:
        want = overhead / (t_def - t_auto)
        got = compute_break_even(overhead, t_def, t_auto)
        err = abs(got - want) / max(abs(want), 1e-8)
        if err > max_err:
            max_err = err
    return {"rel_err": float(max_err)}
