import ref


def check(workdir):
    from partitioner.toy import compute_toy_recompute_set

    model = ref.get_test_module()
    want = ref.compute_reference_recompute_set()
    got = set(compute_toy_recompute_set(model))
    match = 1.0 if got == want else 0.0
    return {"recompute_matched": match}
