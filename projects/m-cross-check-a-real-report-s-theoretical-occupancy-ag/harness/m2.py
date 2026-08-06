import ref


def check(workdir):
    from occupancy.calc import compute_theoretical_occupancy

    ok = 1
    for kernel in ref.KERNELS:
        want = ref.compute_theoretical_occupancy(kernel)
        got = compute_theoretical_occupancy(kernel)
        rel_err = abs(got - want) / (abs(want) + 1e-9)
        if rel_err > 1e-4:
            ok = 0
    out = {"rel_err_ok": float(ok)}
    return out
