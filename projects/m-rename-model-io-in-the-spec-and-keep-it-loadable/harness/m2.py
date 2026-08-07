import ref


def check(workdir):
    from mlspec.blobs import detect_duplicate_blobs
    ok = True
    for b in ref.BLOBS:
        got = detect_duplicate_blobs(b)
        want = ref.detect_duplicate_blobs(b)
        if sorted(got) != sorted(want):
            ok = False
    return {"duplicates_matched": 1.0 if ok else 0.0}
