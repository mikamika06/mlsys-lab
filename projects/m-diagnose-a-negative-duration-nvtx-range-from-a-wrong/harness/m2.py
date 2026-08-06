import ref


def check(workdir):
    from nvdiag.ranking import rank_phases
    _, phases = ref.generate_scenario()
    got = rank_phases(phases)
    sorted_phases = sorted(phases, key=lambda x: x["self"], reverse=True)
    want = [p["name"] for p in sorted_phases[:5]]
    out = {"ranking_matched": 0.0}
    if got == want:
        out["ranking_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
