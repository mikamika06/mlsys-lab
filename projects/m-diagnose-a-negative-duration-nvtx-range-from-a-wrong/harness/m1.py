import ref


def check(workdir):
    from nvtxprof.nvtx import diagnose_nvtx_mismatches

    out = {"mismatches_diagnosed": 0.0, "ranges_matched": 0.0}
    want = ref.diagnose_nvtx_mismatches(ref.NVTX_EVENTS)
    try:
        got = diagnose_nvtx_mismatches(ref.NVTX_EVENTS)
    except Exception as e:
        out["_note"] = f"diagnose_nvtx_mismatches raised {type(e).__name__}: {e}"
        return out

    if got is None or not isinstance(got, dict):
        out["_note"] = "diagnose_nvtx_mismatches returned invalid structure"
        return out

    got_ranges = got.get("ranges", [])
    want_ranges = want.get("ranges", [])
    if got_ranges == want_ranges:
        out["ranges_matched"] = 1.0

    neg_ok = got.get("negative_ranges", []) == want.get("negative_ranges", [])
    unclosed_ok = got.get("unclosed_pushes", []) == want.get("unclosed_pushes", [])
    orphan_ok = got.get("orphan_pops", []) == want.get("orphan_pops", [])

    if neg_ok and unclosed_ok and orphan_ok:
        out["mismatches_diagnosed"] = 1.0
    else:
        out["_note"] = f"mismatch in diagnostics: got {got}, want {want}"

    return out
