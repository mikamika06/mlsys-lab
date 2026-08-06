import ref


def check(workdir):
    from oracle.trace import trace_intermediates
    from oracle.diagnose import diagnose_fp32_regions

    cases = ref.generate_test_cases()
    matched = 0
    total = len(cases)

    for idx, case in enumerate(cases):
        want_trace = ref.reference_trace(case)
        want_diag = ref.reference_diagnose(want_trace)

        try:
            got_trace = trace_intermediates(None, case)
            got_diag = diagnose_fp32_regions(got_trace)
            
            norm_got = [{"out": d["out"], "op": d["op"], "reason": d["reason"]} for d in got_diag]
            norm_want = [{"out": d["out"], "op": d["op"], "reason": d["reason"]} for d in want_diag]

            if norm_got == norm_want:
                matched += 1
            else:
                return {
                    "diagnoses_matched": float(matched / total),
                    "_note": f"Case {idx} diagnosis mismatch. Got {norm_got}, expected {norm_want}"
                }
        except Exception as e:
            return {
                "diagnoses_matched": float(matched / total),
                "_note": f"Exception on case {idx}: {type(e).__name__}: {str(e)}"
            }

    return {"diagnoses_matched": 1.0}
