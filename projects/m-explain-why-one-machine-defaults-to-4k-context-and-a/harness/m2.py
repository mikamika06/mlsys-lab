import ref


def check(workdir):
    from runner.slots import allocate_slots

    out = {"slots_matched": 0.0, "effective_ctx_match": 0.0}
    num_ctx = 32768
    parallel = 4
    want = ref.compute_slots(num_ctx, parallel)
    got = allocate_slots(num_ctx, parallel)
    if got == want:
        out["slots_matched"] = 1.0
        out["effective_ctx_match"] = 1.0
    return out
