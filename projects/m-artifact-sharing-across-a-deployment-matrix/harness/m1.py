import ref


def check(workdir):
    out = {"matrix_resolved": 0.0, "cache_hit_ratio": 0.0}
    try:
        from matrix.artifact import ArtifactRegistry
    except Exception as e:
        out["_note"] = f"Failed to import ArtifactRegistry: {e}"
        return out

    reg = ArtifactRegistry()
    specs = ref.generate_deployment_matrix()

    for i, s in enumerate(specs):
        art_id = f"art_unique_{i}"
        reg.register_artifact(s, art_id)

    matched = 0
    for i, s in enumerate(specs):
        got = reg.resolve_artifact(s)
        if got == f"art_unique_{i}":
            matched += 1

    for s in specs:
        reg.resolve_artifact(s)

    stats = reg.get_cache_stats()

    if matched == len(specs):
        out["matrix_resolved"] = 1.0

    if stats.get("hits", 0) >= len(specs) and stats.get("ratio", 0.0) >= 0.5:
        out["cache_hit_ratio"] = 1.0

    return out
