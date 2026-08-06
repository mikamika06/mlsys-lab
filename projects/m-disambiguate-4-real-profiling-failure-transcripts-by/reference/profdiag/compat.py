def _parse_version(v_str):
    parts = []
    for p in v_str.split('.'):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def resolve_ncu_compatibility(driver_version, ncu_version, compat_table):
    d_ver = _parse_version(driver_version)
    n_ver = _parse_version(ncu_version)

    matched_rule = None
    for rule in compat_table:
        min_d = _parse_version(rule["min_driver"])
        max_d = _parse_version(rule["max_driver"]) if rule.get("max_driver") else (999, 999)
        min_n = _parse_version(rule["min_ncu"])
        max_n = _parse_version(rule["max_ncu"]) if rule.get("max_ncu") else (999, 999)

        if min_d <= d_ver <= max_d and min_n <= n_ver <= max_n:
            matched_rule = rule
            break

    if matched_rule is None:
        return {
            "compatible": False,
            "recommended_ncu": None,
            "action": "UPGRADE_DRIVER_OR_NCU"
        }

    if matched_rule.get("status") == "compatible":
        return {
            "compatible": True,
            "recommended_ncu": ncu_version,
            "action": "OK"
        }
    else:
        return {
            "compatible": False,
            "recommended_ncu": matched_rule.get("fallback_ncu"),
            "action": matched_rule.get("action", "USE_FALLBACK_NCU")
        }
