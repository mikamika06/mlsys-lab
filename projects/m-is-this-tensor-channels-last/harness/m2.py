import ref


def check(workdir):
    out = {"pipeline_matched": 0.0}
    try:
        from layout.pipeline import steady_state_batch_time
    except ImportError:
        return out

    ok = 0
    for case in ref.PIPELINE_CASES:
        cpu, xfer, gpu, pin, nb = case
        want = ref.steady_state_batch_time(cpu, xfer, gpu, pin, nb)
        try:
            if steady_state_batch_time(cpu, xfer, gpu, pin, nb) == want:
                ok += 1
        except Exception:
            pass

    out["pipeline_matched"] = float(ok)
    return out
