import ref


def check(workdir):
    from nsysprof.parser import compute_gpu_idle_gap_pct

    kernels, wall_start, wall_end = ref.generate_nsys_capture()
    expected = ref.compute_idle_gap_ref(kernels, wall_start, wall_end)
    got = compute_gpu_idle_gap_pct(kernels, wall_start, wall_end)

    if expected == 0:
        rel_err = 0.0 if got == 0 else 1.0
    else:
        rel_err = abs(got - expected) / abs(expected)

    return {"rel_err": float(rel_err)}
