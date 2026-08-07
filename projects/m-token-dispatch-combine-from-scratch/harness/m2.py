import ref


def check(workdir):
    from moe.metrics import communication_volume

    out = {"metrics_matched": 0.0}
    num_cases = len(ref.CASES_M2)
    matched = 0

    for idx, case in enumerate(ref.CASES_M2):
        want = ref.reference_communication_volume(**case)
        try:
            got = communication_volume(**case)
            if isinstance(got, dict) and all(k in got for k in ("moe_total_bytes", "dense_total_bytes", "ratio_moe_to_dense")):
                err1 = abs(got["moe_total_bytes"] - want["moe_total_bytes"])
                err2 = abs(got["dense_total_bytes"] - want["dense_total_bytes"])
                err3 = abs(got["ratio_moe_to_dense"] - want["ratio_moe_to_dense"])
                if err1 == 0 and err2 == 0 and err3 < 1e-4:
                    matched += 1
                elif "_note" not in out:
                    out["_note"] = f"case {idx}: got {got}, want {want}"
            elif "_note" not in out:
                out["_note"] = f"case {idx}: invalid return keys/format"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {idx} raised {type(e).__name__}: {str(e)[:100]}"

    out["metrics_matched"] = float(matched / num_cases)
    return out
