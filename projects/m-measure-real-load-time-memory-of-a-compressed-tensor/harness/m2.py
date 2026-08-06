import ref


def check(workdir):
    from compress.measure import simulate_load_memory

    errors = []
    for cfg in ref.CONFIGS:
        want = float(ref.simulate_load_memory(cfg))
        got = float(simulate_load_memory(cfg))
        if want == 0.0:
            err = 0.0 if got == 0.0 else 1.0
        else:
            err = abs(got - want) / abs(want)
        errors.append(err)
    max_err = max(errors) if errors else 1.0
    return {"rel_err": float(max_err)}
