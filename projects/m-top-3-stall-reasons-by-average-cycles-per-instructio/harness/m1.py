import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"top_stalls_matched": 0.0}
    try:
        from warpanalyze.stalls import top_stall_reasons
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {str(e)}"
        return out

    datasets = ref.generate_warp_stats_datasets()
    passed = 0
    total = len(datasets)

    for ds in datasets:
        want = ref.ref_top_stall_reasons(ds, k=3)
        try:
            got = top_stall_reasons(ds, k=3)
        except Exception as e:
            out["_note"] = f"Execution error: {type(e).__name__}: {str(e)}"
            return out

        if len(got) != len(want):
            out["_note"] = f"Length mismatch: got {len(got)}, want {len(want)}"
            return out

        match = True
        for g, w in zip(got, want):
            if g["reason"] != w["reason"] or abs(g["avg_cpi"] - w["avg_cpi"]) > 1e-5:
                match = False
                break
        if match:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"Mismatch: got {got[:2]}, want {want[:2]}"

    if passed == total:
        out["top_stalls_matched"] = 1.0
    return out
