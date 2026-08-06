import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from specdiag.diagnose import diagnose_all_outcomes

    payloads = ref.make_diagnostic_payloads()
    expected = ref.OUTCOMES

    out = {"diagnoses_matched": 0.0}
    try:
        results = diagnose_all_outcomes(payloads)
        matched = sum(1 for got, want in zip(results, expected) if got == want)
        out["diagnoses_matched"] = float(matched)
        if matched < len(expected):
            out["_note"] = f"Expected {expected}, got {results}"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Failed diagnosis check: {type(e).__name__}: {str(e)[:120]}"

    return out
