import ref


def check(workdir):
    from zero1.formula import compute_zero1_memory

    num_params = 10000000
    world_size = 4
    got = compute_zero1_memory(num_params, world_size, 4, "adam")
    want = ref.oracle_compute_memory(num_params, world_size, 4, "adam")

    match = 1.0 if got == want else 0.0
    return {"formulas_matched": match}
