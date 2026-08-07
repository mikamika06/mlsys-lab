def validate_scaled_object(manifest):
    if manifest.get("kind") != "ScaledObject":
        return False
    spec = manifest.get("spec", {})
    min_r = spec.get("minReplicaCount", 0)
    max_r = spec.get("maxReplicaCount", 0)
    if min_r > max_r:
        return False
    triggers = spec.get("triggers", [])
    if not triggers:
        return False
    for t in triggers:
        if t.get("type") == "prometheus":
            meta = t.get("metadata", {})
            if "query" not in meta or "threshold" not in meta:
                return False
    return True
