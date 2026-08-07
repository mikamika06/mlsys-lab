import math
import ref


def check(workdir):
    from hardening.admission import admit_request

    out = {"admission_decisions_matched": 0.0}
    test_cases = [
        (ref.REQUEST_CONFIGS[0], 3.0),
        (ref.REQUEST_CONFIGS[1], 5.0),
        (ref.REQUEST_CONFIGS[3], 1.0),
        (ref.REQUEST_CONFIGS[4], 10.0),
    ]

    matched = 0
    total = len(test_cases)

    for i, (cfg, max_sec) in enumerate(test_cases):
        exp_admit, exp_cost, exp_reason = ref.oracle_admit_request(cfg, ref.PROFILE_PARAMS, max_sec)
        try:
            got_admit, got_cost, got_reason = admit_request(cfg, ref.PROFILE_PARAMS, max_sec)
            same_admit = (got_admit == exp_admit)
            same_reason = (got_reason == exp_reason)
            same_cost = math.isclose(got_cost, exp_cost, rel_tol=1e-5, abs_tol=1e-5)

            if same_admit and same_reason and (same_cost or not exp_admit and exp_cost == 0.0):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Case {i}: expected ({exp_admit}, {exp_cost}, {exp_reason}), got ({got_admit}, {got_cost}, {got_reason})"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Case {i} raised {type(e).__name__}: {e}"

    if matched == total:
        out["admission_decisions_matched"] = 1.0

    return out
