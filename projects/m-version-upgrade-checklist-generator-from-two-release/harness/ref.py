RAW_RELEASE_NOTES = [
    """
FLAG: gpu_memory_utilization = 0.90
FLAG: enforce_eager = False
CONFIG: max_model_len = 2048
CONFIG: block_size = 16
""",
    """
FLAG: gpu_memory_utilization = 0.95
FLAG: enforce_eager = False
FLAG: kv_cache_dtype = fp8
CONFIG: max_model_len = 4096
CONFIG: block_size = 16
DEPRECATED: legacy_ray_launcher
""",
    """
FLAG: gpu_memory_utilization = 0.95
FLAG: kv_cache_dtype = fp8
CONFIG: max_model_len = 8192
CONFIG: block_size = 32
DEPRECATED: legacy_ray_launcher
BREAKING: remove --enforce_eager legacy alias
""",
    """
FLAG: gpu_memory_utilization = 0.90
CONFIG: max_model_len = 16384
CONFIG: block_size = 32
DEPRECATED: legacy_ray_launcher
DEPRECATED: use_v2_block_manager
BREAKING: remove --enforce_eager legacy alias
BREAKING: require CUDA 12.2+
""",
    """
FLAG: gpu_memory_utilization = 0.90
FLAG: enable_chunked_prefill = True
CONFIG: max_model_len = 16384
CONFIG: block_size = 32
DEPRECATED: legacy_ray_launcher
DEPRECATED: use_v2_block_manager
BREAKING: require CUDA 12.2+
"""
]


def parse_release_notes(text):
    flags = {}
    configs = {}
    deprecations = set()
    breaking = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("FLAG:"):
            parts = line[5:].strip().split("=")
            if len(parts) == 2:
                flags[parts[0].strip()] = parts[1].strip()
        elif line.startswith("CONFIG:"):
            parts = line[7:].strip().split("=")
            if len(parts) == 2:
                configs[parts[0].strip()] = parts[1].strip()
        elif line.startswith("DEPRECATED:"):
            item = line[11:].strip()
            if item:
                deprecations.add(item)
        elif line.startswith("BREAKING:"):
            item = line[9:].strip()
            if item:
                breaking.add(item)

    return {
        "flags": flags,
        "configs": configs,
        "deprecations": deprecations,
        "breaking": breaking,
    }


def compute_upgrade_diff(old_snap, new_snap):
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


def generate_checklist(upgrade_diff):
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
