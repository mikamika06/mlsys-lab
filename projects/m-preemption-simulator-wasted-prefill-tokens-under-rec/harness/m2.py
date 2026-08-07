import ref


def check(workdir):
    from preemption.model import find_breakeven_point
    p1, p2, p3, expected = ref.get_m2_data()
    got = find_breakeven_point(p1, p2, p3)
    if expected == 0:
        rel_err = 0.0 if got == 0 else 1.0
    else:
        rel_err = abs(got - expected) / abs(expected)
    return {"rel_err": float(rel_err)}
