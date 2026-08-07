import ref

def check(workdir):
    m = {"diff_detected": 0.0}
    try:
        from trt_builder.inspector import inspect_diff
        ea, eb = ref.get_sample_engines()
        eb_modified = dict(eb)
        eb_modified["weights_hash"] = "hash_v2"
        diffs = inspect_diff(ea, eb_modified)
        if "weights_hash" in diffs:
            m["diff_detected"] = 1.0
    except Exception:
        pass
    return m
