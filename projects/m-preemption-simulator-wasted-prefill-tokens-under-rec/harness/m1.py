import ref


def check(workdir):
    from preemption.simulator import simulate_wasted_tokens
    reqs, preempts, expected = ref.get_m1_data()
    got = simulate_wasted_tokens(reqs, preempts)
    if expected == 0:
        rel_err = 0.0 if got == 0 else 1.0
    else:
        rel_err = abs(got - expected) / abs(expected)
    return {"rel_err": float(rel_err)}
