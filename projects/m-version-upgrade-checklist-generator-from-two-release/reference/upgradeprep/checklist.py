def generate_checklist(upgrade_diff):
    """Generate prioritized actionable checklist items from an upgrade diff."""
    items = []

    for item in upgrade_diff.get("new_breaking", []):
        items.append({"priority": "CRITICAL", "category": "BREAKING", "action": f"Fix breaking change: {item}"})

    for flag in upgrade_diff.get("removed_flags", []):
        items.append({"priority": "HIGH", "category": "FLAG_REMOVED", "action": f"Remove deprecated CLI flag '--{flag}'"})

    for flag, (old_v, new_v) in upgrade_diff.get("changed_flags", {}).items():
        items.append({"priority": "MEDIUM", "category": "FLAG_CHANGED", "action": f"Update flag '--{flag}' default from '{old_v}' to '{new_v}'"})

    for cfg, (old_v, new_v) in upgrade_diff.get("changed_configs", {}).items():
        items.append({"priority": "MEDIUM", "category": "CONFIG_CHANGED", "action": f"Verify config '{cfg}' change: '{old_v}' -> '{new_v}'"})

    for dep in upgrade_diff.get("new_deprecations", []):
        items.append({"priority": "LOW", "category": "DEPRECATED", "action": f"Plan migration for deprecated feature: {dep}"})

    return items
