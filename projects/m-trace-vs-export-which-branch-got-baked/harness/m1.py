import ref


def check(workdir):
    from gcapture.branch import inspect_baked_branch

    out = {"baked_branches_detected": 0.0}
    mod, inp1, inp2 = ref.get_branch_test_setup()

    want = ref.ref_inspect_baked_branch(mod, inp1, inp2)
    got = inspect_baked_branch(mod, inp1, inp2)

    if got == want:
        out["baked_branches_detected"] = 1.0
    else:
        out["_note"] = f"Expected {want}, got {got}"

    return out
