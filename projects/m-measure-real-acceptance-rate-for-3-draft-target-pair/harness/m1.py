import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from specpair.metrics import evaluate_pairings
    from specpair.tokenizer import classify_tokenizer_compatibility

    out = {"acceptance_rate_matched": 0.0, "compatibility_matched": 0.0}

    ref_rates = ref.evaluate_pairings(ref.PAIRINGS_DATA)
    try:
        got_rates = evaluate_pairings(ref.PAIRINGS_DATA)
        rates_ok = True
        for k, v in ref_rates.items():
            if k not in got_rates or abs(got_rates[k] - v) > 1e-6:
                rates_ok = False
                out["_note"] = f"mismatch on pairing {k}: got {got_rates.get(k)}, want {v}"
                break
        if rates_ok:
            out["acceptance_rate_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"evaluate_pairings raised {type(e).__name__}: {str(e)}"
        return out

    compat_ok = True
    for case in ref.TOKENIZER_CASES:
        try:
            got_compat = classify_tokenizer_compatibility(case["draft"], case["target"])
            if got_compat != case["expected"]:
                compat_ok = False
                out["_note"] = (
                    f"compat mismatch: got {got_compat}, want {case['expected']}"
                )
                break
        except Exception as e:
            compat_ok = False
            out["_note"] = (
                f"classify_tokenizer_compatibility raised {type(e).__name__}: {str(e)}"
            )
            break

    if compat_ok:
        out["compatibility_matched"] = 1.0

    return out
