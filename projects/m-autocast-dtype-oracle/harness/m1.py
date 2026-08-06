import ref


def check(workdir):
    from oracle.trace import trace_intermediates

    cases = ref.generate_test_cases()
    matched = 0
    total = len(cases)

    for idx, case in enumerate(cases):
        want = ref.reference_trace(case)
        try:
            got = trace_intermediates(None, case)
            if got.get("intermediates") == want.get("intermediates"):
                matched += 1
            else:
                return {
                    "maps_matched": float(matched / total),
                    "_note": f"Case {idx} output mismatch. Got {got}, expected {want}"
                }
        except Exception as e:
            return {
                "maps_matched": float(matched / total),
                "_note": f"Exception raised on case {idx}: {type(e).__name__}: {str(e)}"
            }

    return {"maps_matched": 1.0}
