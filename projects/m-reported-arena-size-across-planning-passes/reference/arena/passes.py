def track_arena_sizes(passes):
    sizes = []
    for p in passes:
        sizes.append(int(p.get("arena_size", 0)))
    converged = len(set(sizes[-2:])) == 1 if len(sizes) >= 2 else True
    return {"sizes": sizes, "converged": converged, "max_size": max(sizes) if sizes else 0}
