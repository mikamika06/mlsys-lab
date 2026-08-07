from typing import List, Dict, Any


def compute_resident_versions(available_versions: List[int], policy: Dict[str, Any]) -> List[int]:
    valid_versions = sorted([v for v in available_versions if isinstance(v, int) and v > 0])
    if not valid_versions:
        return []

    kind = policy.get("kind", "all")

    if kind == "all":
        return valid_versions
    elif kind == "latest":
        count = policy.get("count", 1)
        if count <= 0:
            return []
        return valid_versions[-count:]
    elif kind == "specific":
        target_versions = set(policy.get("versions", []))
        return [v for v in valid_versions if v in target_versions]

    return valid_versions
