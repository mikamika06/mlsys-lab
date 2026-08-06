import ref


def check(workdir):
    from pareto.front import select_pareto_front

    out = {"front_matched": 0.0}
    pts = ref.SAMPLE_POINTS
    want = ref.select_pareto_front(pts)
    try:
        got = select_pareto_front(pts)
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {str(e)[:100]}"
        return out

    norm_want = sorted([p["id"] for p in want])
    try:
        norm_got = sorted([p["id"] for p in (got or [])])
    except Exception:
        norm_got = []

    if norm_got == norm_want:
        out["front_matched"] = 1.0
    else:
        out["_note"] = f"got front ids {norm_got}, expected {norm_want}"
    return out
