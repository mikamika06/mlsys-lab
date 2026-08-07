def compute_resident_versions(available_versions: list, policy: dict) -> list:
    sorted_versions = sorted(int(v) for v in available_versions)
    if not sorted_versions:
        return []

    if "all" in policy:
        return sorted_versions
    elif "latest" in policy:
        count = int(policy["latest"].get("count", 1))
        if count <= 0:
            return []
        return sorted_versions[-count:]
    elif "specific" in policy:
        versions_to_keep = [int(v) for v in policy["specific"].get("versions", [])]
        return sorted([v for v in versions_to_keep if v in sorted_versions])
    else:
        return [sorted_versions[-1]]
