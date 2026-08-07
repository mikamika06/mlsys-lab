import ref


def check(workdir):
    from batching.sweep import sweep_parameters
    out = {"sweep_matched": 0.0}
    want = ref.sweep_parameters(ref.SIZES, ref.TIMEOUTS, ref.WORKLOAD)
    try:
        got = sweep_parameters(ref.SIZES, ref.TIMEOUTS, ref.WORKLOAD)
        if len(got) == len(want):
            match = True
            for g, w in zip(got, want):
                if g.get("max_batch_size") != w["max_batch_size"] or g.get("batch_wait_timeout_s") != w["batch_wait_timeout_s"]:
                    match = False
            if match:
                out["sweep_matched"] = 1.0
    except Exception:
        pass
    return out
