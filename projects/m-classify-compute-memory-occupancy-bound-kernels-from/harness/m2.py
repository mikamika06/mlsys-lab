import ref


def check(workdir):
    from profiler.analysis import compute_proton_breakdown

    lines, expected = ref.get_proton_sample()
    out = {"regions_matched": 0.0}
    try:
        got = compute_proton_breakdown(lines)
        if got == expected:
            out["regions_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, expected {expected}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:100]}"
    return out
