def compute_upgrade_diff(old_snap, new_snap):
    """Compute structured differences between two release snapshots."""
    removed_flags = sorted(list(set(old_snap["flags"].keys()) - set(new_snap["flags"].keys())))
    added_flags = sorted(list(set(new_snap["flags"].keys()) - set(old_snap["flags"].keys())))
    changed_flags = {}
    for k in old_snap["flags"]:
        if k in new_snap["flags"] and old_snap["flags"][k] != new_snap["flags"][k]:
            changed_flags[k] = (old_snap["flags"][k], new_snap["flags"][k])

    changed_configs = {}
    all_configs = set(old_snap["configs"].keys()) | set(new_snap["configs"].keys())
    for k in sorted(list(all_configs)):
        old_val = old_snap["configs"].get(k)
        new_val = new_snap["configs"].get(k)
        if old_val != new_val:
            changed_configs[k] = (old_val, new_val)

    new_deprecations = sorted(list(new_snap["deprecations"] - old_snap["deprecations"]))
    new_breaking = sorted(list(new_snap["breaking"]))

    return {
        "removed_flags": removed_flags,
        "added_flags": added_flags,
        "changed_flags": changed_flags,
        "changed_configs": changed_configs,
        "new_deprecations": new_deprecations,
        "new_breaking": new_breaking,
    }
