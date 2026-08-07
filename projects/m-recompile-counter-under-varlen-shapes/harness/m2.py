import ref


def check(workdir):
    from recompile.bucketing import bucket_shape
    from recompile.capture import capture_decode_step

    out = {"bucketing_match": 0.0, "capture_match": 0.0}

    ok_bucket = True
    for s in [10, 64, 65, 200, 600]:
        want = ref.bucket_shape(s, ref.BUCKETS)
        got = bucket_shape(s, ref.BUCKETS)
        if got != want:
            ok_bucket = False

    if ok_bucket:
        out["bucketing_match"] = 1.0

    try:
        def dummy(x):
            return x + 1
        graph = capture_decode_step(dummy, (10,))
        res = graph.replay()
        if res == 11:
            out["capture_match"] = 1.0
    except Exception as e:
        out["_note"] = f"capture error: {e}"

    return out
