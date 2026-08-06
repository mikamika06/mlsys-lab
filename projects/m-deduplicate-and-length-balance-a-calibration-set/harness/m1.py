import ref


def check(workdir):
    out = {"dedup_matched": 0.0}
    try:
        from calib.dedup import deduplicate_samples

        dataset = ref.generate_dataset()
        want = ref.ref_deduplicate_samples(dataset, num_perm=128, threshold=0.8)
        got = deduplicate_samples(dataset, num_perm=128, threshold=0.8)

        if len(got) == len(want):
            out["dedup_matched"] = 1.0
        else:
            out["_note"] = f"Expected {len(want)} samples post-dedup, got {len(got)}"
    except Exception as e:
        out["_note"] = f"Failed with exception: {str(e)}"
    return out
