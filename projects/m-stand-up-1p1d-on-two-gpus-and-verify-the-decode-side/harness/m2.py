import ref


def check(workdir):
    try:
        from disagg.p1d import DecodeWorker, Pipeline1P1D, PrefillWorker
        from disagg.verify import analyze_execution_metrics, verify_decode_skips_prefill
    except Exception as e:
        return {"decode_prefill_ratio": 1.0, "verification_passed": 0.0, "_note": f"Import failure: {e}"}

    pw = PrefillWorker(0, 8, 64, 8)
    dw = DecodeWorker(1, 8, 64, 8)
    pipe = Pipeline1P1D(pw, dw)

    ratios = []
    verified_all = True

    for req in ref.TEST_REQUESTS:
        try:
            res = pipe.process_request(req["id"], req["prompt"], req["steps"])
            ver = verify_decode_skips_prefill(res)
            analysis = ver.get("analysis", {})
            ratios.append(analysis.get("decode_prefill_ratio", 1.0))
            if not ver.get("verified", False):
                verified_all = False
        except Exception as e:
            return {"decode_prefill_ratio": 1.0, "verification_passed": 0.0, "_note": f"Error during verification: {e}"}

    max_ratio = max(ratios) if ratios else 1.0
    return {
        "decode_prefill_ratio": float(max_ratio),
        "verification_passed": float(verified_all)
    }
