import ref


def check(workdir):
    from conv.classifier import classify_error
    from conv.success import compute_success_rate

    errors = [err for m in ref.MODELS for err in m["errors"]]
    err_ok = 0
    for i, err in enumerate(errors):
        if classify_error(err) == ref.classify_error(err):
            err_ok += 1

    results = [{"success": True}, {"success": False}, {"success": True}]
    got_rate = compute_success_rate(results)
    want_rate = ref.compute_success_rate(results)

    out = {
        "errors_classified": float(err_ok),
        "success_rate_matched": 1.0 if got_rate == want_rate else 0.0
    }
    return out
