import ref


def check(workdir):
    from profiler.core import top_kernels

    try:
        got = top_kernels(ref.KERNEL_ROWS, n=3)
    except Exception as e:
        return {"top_kernels_match": 0.0, "_note": f"raised exception: {e}"}

    # Compute reference top kernels
    totals = {}
    for k in ref.KERNEL_ROWS:
        name = k["name"]
        dur = k["end"] - k["start"]
        totals[name] = totals.get(name, 0.0) + dur
    grand_total = sum(totals.values())
    sorted_k = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:3]
    want = []
    for name, t in sorted_k:
        share = (t / grand_total) if grand_total > 0 else 0.0
        want.append({"name": name, "total_time": t, "share": share})

    if not isinstance(got, list) or len(got) != len(want):
        return {"top_kernels_match": 0.0, "_note": f"got length {len(got) if isinstance(got, list) else type(got)}, want {len(want)}"}

    match = True
    for g, w in zip(got, want):
        if g.get("name") != w["name"] or abs(g.get("total_time", -1) - w["total_time"]) > 1e-5 or abs(g.get("share", -1) - w["share"]) > 1e-5:
            match = False
            break

    score = 1.0 if match else 0.0
    out = {"top_kernels_match": score}
    if score == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
