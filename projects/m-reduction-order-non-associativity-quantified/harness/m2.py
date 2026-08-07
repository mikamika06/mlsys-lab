import ref


def check(workdir):
    from redscale.outliers import detect_loss_spike_ranks
    out = {"spike_match": 0.0}
    try:
        ok = 0
        for mat in ref.TEST_MATRICES:
            got = detect_loss_spike_ranks(mat)
            want = ref.detect_loss_spike_ranks(mat)
            if set(got) == set(want):
                ok += 1
        out["spike_match"] = 1.0 if ok == len(ref.TEST_MATRICES) else 0.0
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    return out
