def compute_os_floor(variant, feature_os_map):
    req_os = variant.get("min_os", (0, 0))
    for feat in variant.get("required_features", []):
        if feat in feature_os_map:
            feat_os = feature_os_map[feat]
            if feat_os > req_os:
                req_os = feat_os
    return req_os


def filter_eligible_devices(devices, variant, feature_os_map):
    os_floor = compute_os_floor(variant, feature_os_map)
    eligible = []
    excluded = {}
    for dev in devices:
        dev_id = dev["id"]
        if dev["os_version"] < os_floor:
            excluded[dev_id] = "os_below_floor"
            continue
        missing_feat = [
            f for f in variant.get("required_features", []) if f not in dev["features"]
        ]
        if missing_feat:
            excluded[dev_id] = "missing_feature"
            continue
        if dev["ram_mb"] < variant.get("min_ram_mb", 0):
            excluded[dev_id] = "insufficient_ram"
            continue
        eligible.append(dev_id)
    return {"eligible": sorted(eligible), "excluded": excluded}
