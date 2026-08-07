import ref


def check(workdir):
    from cnnprune.toy import propagate_channels
    out = {"dep_matched": 0.0}
    pruned = [1, 3, 5]
    try:
        got = propagate_channels(pruned)
        want = ref.get_expected_toy(pruned)
        if got == want:
            out["dep_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {str(e)[:100]}"
    return out
