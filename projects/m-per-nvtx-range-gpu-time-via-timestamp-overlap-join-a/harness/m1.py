import ref


def check(workdir):
    from profiler.core import compute_nvtx_gpu_time

    try:
        got = compute_nvtx_gpu_time(ref.NVTX_ROWS, ref.KERNEL_ROWS)
    except Exception as e:
        return {"ranges_matched": 0.0, "_note": f"raised exception: {e}"}

    if not isinstance(got, list) or len(got) != len(ref.NVTX_ROWS):
        return {"ranges_matched": 0.0, "_note": f"expected list of length {len(ref.NVTX_ROWS)}, got {type(got)}"}

    # Compute expected reference output
    want = []
    for r in ref.NVTX_ROWS:
        r_start, r_end, r_name = r["start"], r["end"], r["name"]
        tot = 0.0
        for k in ref.KERNEL_ROWS:
            os_ = max(r_start, k["start"])
            oe_ = min(r_end, k["end"])
            if os_ < oe_:
                tot += (oe_ - os_)
        want.append({"name": r_name, "gpu_time": tot})

    match_count = 0
    for g, w in zip(got, want):
        if g.get("name") == w["name"] and abs(g.get("gpu_time", -1) - w["gpu_time"]) < 1e-5:
            match_count += 1

    score = 1.0 if match_count == len(ref.NVTX_ROWS) else 0.0
    out = {"ranges_matched": score}
    if score == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
