import ref


def check(workdir):
    out = {"balance_matched": 0.0}
    try:
        from calib.balance import balance_lengths

        dataset = ref.generate_dataset()
        buckets = [30, 60, 100]
        targets = {30: 10, 60: 15, 100: 5}

        want = ref.ref_balance_lengths(dataset, buckets, targets)
        got = balance_lengths(dataset, buckets, targets)

        if len(got) == len(want):
            matching_lengths = sum(1 for g, w in zip(got, want) if len(g) == len(w))
            if matching_lengths == len(want):
                out["balance_matched"] = 1.0
            else:
                out["_note"] = "Sample lengths do not match expected bucket distributions"
        else:
            out["_note"] = f"Expected length {len(want)}, got {len(got)}"
    except Exception as e:
        out["_note"] = f"Failed with exception: {str(e)}"
    return out
