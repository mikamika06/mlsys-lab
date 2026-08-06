import ref


def check(workdir):
    from lorascaling.extractor import extract_rank_scaling

    out = {"logs_matched": 0.0}
    ok = 0
    for i, (log1, log2, _) in enumerate(ref.LOG_PAIRS):
        want = ref.oracle_extract(log1, log2)
        try:
            got = extract_rank_scaling(log1, log2)
            match = True
            for k in ["vram_base", "vram_slope", "flops_base", "flops_slope"]:
                if abs(got[k] - want[k]) > 1e-4 * (abs(want[k]) + 1.0):
                    match = False
                    break
            if match:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"pair {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"pair {i} raised exception: {type(e).__name__}: {str(e)}"

    out["logs_matched"] = float(ok)
    return out
