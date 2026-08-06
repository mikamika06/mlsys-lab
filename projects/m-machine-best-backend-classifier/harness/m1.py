import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    sys.path.insert(0, workdir)
    from fa_backend.classifier import classify_backend
    from fa_backend.failure import explain_platform_failure

    out = {"backends_matched": 0.0, "failures_matched": 0.0}

    total_combos = float(len(ref.MACHINES) * len(ref.INPUT_SPECS))
    cls_ok = 0
    for m in ref.MACHINES:
        for spec in ref.INPUT_SPECS:
            want = ref.ref_classify(m, spec)
            got = classify_backend(m, spec)
            if got == want:
                cls_ok += 1
            elif "_note" not in out:
                out["_note"] = f"classify mismatch on {m['id']}/{spec['id']}: got {got}, want {want}"

    out["backends_matched"] = float(cls_ok == total_combos)

    fail_ok = 0
    total_failures = float(len(ref.MACHINES) * len(ref.INPUT_SPECS) * len(ref.TARGET_BACKENDS))
    for m in ref.MACHINES:
        for spec in ref.INPUT_SPECS:
            for b in ref.TARGET_BACKENDS:
                want_exp = ref.ref_explain_failure(b, m, spec)
                got_exp = explain_platform_failure(b, m, spec)
                got_set = set(got_exp.split("|")) if got_exp != "NO_FAILURE" else {"NO_FAILURE"}
                want_set = set(want_exp.split("|")) if want_exp != "NO_FAILURE" else {"NO_FAILURE"}
                if got_set == want_set:
                    fail_ok += 1
                elif "_note" not in out:
                    out["_note"] = f"explanation mismatch on {b} for {m['id']}/{spec['id']}: got {got_exp}, want {want_exp}"

    out["failures_matched"] = float(fail_ok == total_failures)
    return out
