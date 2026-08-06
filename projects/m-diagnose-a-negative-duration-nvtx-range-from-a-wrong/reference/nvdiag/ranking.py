def rank_phases(phases):
    sorted_p = sorted(phases, key=lambda x: x["self"], reverse=True)
    return [p["name"] for p in sorted_p[:5]]
