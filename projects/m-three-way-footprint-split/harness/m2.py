import ref


def check(workdir):
    from footprint.selective import selective_registration_win
    from footprint.predictor import predict_peak_rss

    out = {"savings_matched": 0.0, "rss_matched": 0.0}

    kernels, used = ref.SELECTIVE_TEST_CASE
    want_savings = ref.selective_registration_win(kernels, used)
    got_savings = selective_registration_win(kernels, used)

    if got_savings == want_savings:
        out["savings_matched"] = 1.0
    else:
        out["_note"] = f"selective: got {got_savings}, reference {want_savings}"

    plan, align, overhead = ref.PREDICTOR_TEST_CASE
    want_rss = ref.predict_peak_rss(plan, align, overhead)
    got_rss = predict_peak_rss(plan, align, overhead)

    if got_rss == want_rss:
        out["rss_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"predictor: got {got_rss}, reference {want_rss}"

    return out
