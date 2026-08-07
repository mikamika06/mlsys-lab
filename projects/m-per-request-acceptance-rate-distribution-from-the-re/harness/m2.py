import ref


def check(workdir):
    from eagle_diag.config import diagnose_speculative_config

    out = {"configs_diagnosed": 0.0}
    correct = 0
    total = len(ref.DIAGNOSTIC_CONFIG_RECORDS)

    for rec in ref.DIAGNOSTIC_CONFIG_RECORDS:
        want = rec["expected_outcome"]
        try:
            got = diagnose_speculative_config(rec)
            if got == want:
                correct += 1
            elif "_note" not in out:
                out["_note"] = f"Config {rec['config_id']}: got {got}, expected {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Config {rec['config_id']} raised {type(e).__name__}: {str(e)[:100]}"

    if correct == total:
        out["configs_diagnosed"] = 1.0
    return out
