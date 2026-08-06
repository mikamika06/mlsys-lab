import ref

def check(workdir):
    from memdiag.checkpoint import measure_checkpoint_reduction
    from memdiag.allocator import compute_split_fraction

    ckpt_fixtures = ref.get_checkpoint_fixtures()
    ckpt_match = 0
    for fix in ckpt_fixtures:
        got = measure_checkpoint_reduction(fix["layers"], fix["batch_size"])
        if got == fix["expected"]:
            ckpt_match += 1

    alloc_fixtures = ref.get_allocator_fixtures()
    alloc_match = 0
    for fix in alloc_fixtures:
        got = compute_split_fraction(fix["events"])
        if abs(got - fix["expected"]) < 1e-5:
            alloc_match += 1

    return {
        "savings_matched": 1.0 if ckpt_match == len(ckpt_fixtures) else 0.0,
        "fraction_matched": 1.0 if alloc_match == len(alloc_fixtures) else 0.0
    }
